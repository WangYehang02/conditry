import math
import os
import time
import tqdm
import torch
import torch.nn as nn
from typing import Optional, Tuple

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import auc, precision_recall_curve
from pygod.metric.metric import (
    eval_roc_auc,
    eval_average_precision,
    eval_recall_at_k,
    eval_precision_at_k,
)

from pygod.utils import load_data

from fmgad.graph_ops import add_virtual_knn_edges, smooth_scores_by_graph
from fmgad.losses import conditional_flow_matching_loss, flow_matching_loss
from fmgad.models import (
    DiffusionModel,
    FlowMatchingModel,
    GraphAE,
    MLPDiffusion,
    MLPFlowMatching,
    MLPFlowMatchingTwoHead,
    sample_dm,
    sample_flow_matching,
    sample_flow_matching_free,
    sample_flow_matching_pair,
)
from fmgad.residuals import compute_dual_residuals_with_degree
from fmgad.scoring import calibrate_polarity_consensus_rank, compute_local_prior, softmax_with_temperature


class _GateParams(nn.Module):
    """Learnable gate for fusing local vs global residual by node degree."""

    def __init__(self, bias: float = 2.0, sharpness: float = 1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self._raw_sharpness = nn.Parameter(torch.tensor(sharpness, dtype=torch.float32))

    @property
    def sharpness(self):
        return torch.nn.functional.softplus(self._raw_sharpness)


class ResFlowGAD(BaseTransform):
    """Graph anomaly detection via AE latent, dual residual, flow matching, reconstruction error."""

    def __init__(
        self,
        name: str = "FMGAD",
        hid_dim: Optional[int] = None,
        ae_epochs: int = 300,
        diff_epochs: int = 800,
        patience: int = 100,
        lr: float = 0.005,
        wd: float = 0.0,
        weight: float = 1.0,
        sample_steps: int = 1,
        ae_dropout: float = 0.3,
        ae_lr: float = 0.01,
        ae_alpha: float = 0.8,
        use_proto: bool = True,
        profile_efficiency: bool = False,
        proto_alpha: float = 0.01,
        residual_scale: float = 10.0,
        gate_bias: float = 2.0,
        gate_sharpness: float = 1.0,
        verbose: bool = True,
        use_virtual_neighbors: bool = True,
        virtual_degree_threshold: int = 5,
        virtual_k: int = 5,
        score_smoothing_alpha: float = 0.3,
        ensemble_score: bool = True,
        num_trial: int = 3,
        exp_tag: Optional[str] = None,
        polarity_enabled: bool = True,
        polarity_consensus_threshold: float = 0.70,
        polarity_consensus_score_weight: float = 0.90,
        generative_backend: str = "flow",
        proto_normal_temp: float = 50.0,
        use_proto_normal_weight: bool = False,
        dual_flow_mode: Optional[str] = None,
        flow_hidden_dim: int = 512,
        dom_weight_mode: str = "learned",
        dom_kappa_q: float = 0.5,
        dom_tau_q: float = 0.2,
        dom_control_seed: int = 2027,
        allow_center_fallback: bool = False,
    ):
        self.name = name
        self.num_trial = num_trial
        self.hid_dim = hid_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.weight = weight
        self.sample_steps = sample_steps
        self.verbose = verbose
        backend = str(generative_backend or "flow").strip().lower()
        if backend not in ("flow", "diffusion"):
            raise ValueError(f"Unsupported generative_backend={generative_backend!r}")
        self.generative_backend = backend
        mode = None if dual_flow_mode in (None, "", "none") else str(dual_flow_mode).strip().lower()
        allowed = {
            None,
            "single_full",
            "single_ref",
            "duplicate_dual",
            "residual_contrastive",
            "wider_single",
            "shared_two_heads",
            "center_dom",
        }
        if mode not in allowed:
            raise ValueError(f"Unsupported dual_flow_mode={dual_flow_mode!r}")
        self.dual_flow_mode = mode
        self.flow_hidden_dim = int(flow_hidden_dim)
        wmode = str(dom_weight_mode or "learned").strip().lower()
        if wmode not in ("uniform", "learned", "shuffled", "reversed"):
            raise ValueError(f"Unsupported dom_weight_mode={dom_weight_mode!r}")
        self.dom_weight_mode = wmode
        self.dom_kappa_q = float(dom_kappa_q)
        self.dom_tau_q = float(dom_tau_q)
        self.dom_control_seed = int(dom_control_seed)
        self.allow_center_fallback = bool(allow_center_fallback)
        self._last_n_eff = float("nan")
        self._last_n_eff_ratio = float("nan")
        self._last_dom_weight_stats = None
        self._cached_center_latents = None
        # Diffusion ablation: free-only EDM; never use prototype guidance.
        # Dual-flow controls replace free/proto pairing.
        self.use_proto = bool(use_proto) and backend == "flow" and mode is None
        self.profile_efficiency = bool(profile_efficiency)
        self.proto_alpha = proto_alpha
        # Disabled by default: normality-weighted proto CFM hurt Avg AUROC.
        self.proto_normal_temp = float(proto_normal_temp)
        self.use_proto_normal_weight = bool(use_proto_normal_weight)
        self.residual_scale = residual_scale
        self.dm_ref = None  # type: Optional[FlowMatchingModel]
        self.dm_shared = None  # type: Optional[MLPFlowMatchingTwoHead]
        self.gate_module = _GateParams(bias=gate_bias, sharpness=gate_sharpness)
        self.use_virtual_neighbors = use_virtual_neighbors
        self.virtual_degree_threshold = virtual_degree_threshold
        self.virtual_k = virtual_k
        # Score smoothing is always on; alpha=0 makes it a no-op (for ablations).
        self.score_smoothing_alpha = float(score_smoothing_alpha)
        self.ensemble_score = ensemble_score
        self.exp_tag = exp_tag
        self.polarity_enabled = bool(polarity_enabled)
        self.polarity_consensus_threshold = float(polarity_consensus_threshold)
        self.polarity_consensus_score_weight = float(polarity_consensus_score_weight)
        self._local_prior_probe = None  # type: Optional[torch.Tensor]
        self._last_polarity_diag = None

        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha

        self.ae = None  # type: Optional[GraphAE]
        self.dm = None  # type: Optional[FlowMatchingModel]
        self.dm_proto = None  # type: Optional[FlowMatchingModel]
        self.proto = None  # type: Optional[torch.Tensor]

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 100

    def _apply_score_polarity_adapter(self, score: torch.Tensor) -> torch.Tensor:
        if not self.polarity_enabled or self._local_prior_probe is None:
            return score
        probes = [self._local_prior_probe.to(score.device)]
        out, _flipped, diag = calibrate_polarity_consensus_rank(
            score,
            probes,
            agreement_threshold=self.polarity_consensus_threshold,
            score_weight=self.polarity_consensus_score_weight,
        )
        self._last_polarity_diag = diag
        return out

    def _load_dataset(self, dset: str):
        """PyGOD graph anomaly datasets: books / disney / enron / reddit / weibo."""
        return load_data(dset)

    def _ensure_save_dir(self, dset: str):
        # Default checkpoints under cwd/models; set FMGAD_MODEL_ROOT to redirect to a large disk.
        model_root = os.environ.get("FMGAD_MODEL_ROOT", os.path.join(os.getcwd(), "models"))
        save_dir = os.path.join(model_root, dset, "full_batch")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _build_z(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return z = [h; scaled fused residual], h, and fused residual."""
        h = self.ae.encode(x, edge_index)
        dev = h.device
        if self.use_virtual_neighbors and getattr(self, "virtual_degree_threshold", 5) is not None:
            edge_index = add_virtual_knn_edges(
                edge_index, h,
                self.virtual_degree_threshold,
                getattr(self, "virtual_k", 5),
                dev,
            )

        r_global, r_local, deg = compute_dual_residuals_with_degree(h, edge_index)
        bias = self.gate_module.bias.to(dev)
        sharpness = self.gate_module.sharpness.to(dev)
        alpha = torch.sigmoid((deg - bias) * sharpness)
        r_fused = alpha * r_local + (1.0 - alpha) * r_global
        r_final = r_fused * self.residual_scale
        z = torch.cat([h, r_final], dim=1)
        return z, h, r_final

    def _build_z_pair(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Full latent z and residual-suppressed reference z_ref=[h; 0]."""
        z, h, r_final = self._build_z(x, edge_index)
        z_ref = torch.cat([h, torch.zeros_like(r_final)], dim=1)
        return z, z_ref, h, r_final

    def _center_consistency_weights(
        self,
        h: torch.Tensor,
        mode: Optional[str] = None,
        y: Optional[torch.Tensor] = None,
    ):
        """Strict center-consistent weights: s_i = cos(h_i, c) only (no residual fallback).

        q_i = sg[σ((R(s)_i - κ_q) / τ_q)], π_i = q_i / Σ_j q_j.
        Modes: uniform | learned | shuffled | reversed.
        shuffled uses an independent Generator (does not touch global RNG).
        """
        mode = (mode or self.dom_weight_mode).strip().lower()
        n = int(h.shape[0])
        device = h.device
        dtype = h.dtype
        center_valid = True
        feat_used = "h_center"
        s = None

        if mode == "uniform":
            q = torch.ones(n, device=device, dtype=dtype)
            feat_used = "uniform"
        else:
            use = h.detach()
            c = use.mean(dim=0, keepdim=True)
            s = torch.nn.functional.cosine_similarity(use, c.expand(n, -1), dim=1)
            s_std = float(s.std())
            if s_std < 1e-4:
                # Center similarity has no discrimination — mark invalid, do NOT
                # silently switch to residual weighting.
                q = torch.ones(n, device=device, dtype=dtype)
                feat_used = "center_collapsed_uniform"
                center_valid = False
            else:
                order = torch.argsort(s)
                ranks = torch.empty(n, device=device, dtype=dtype)
                ranks[order] = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
                kappa = float(self.dom_kappa_q)
                tau = max(float(self.dom_tau_q), 1e-6)
                q = torch.sigmoid((ranks - kappa) / tau).detach()
                feat_used = "h_center"

            if mode == "shuffled":
                # Independent CPU generator: does not consume the global RNG stream.
                gen = torch.Generator()
                gen.manual_seed(int(self.dom_control_seed))
                perm = torch.randperm(n, generator=gen).to(device)
                q = q[perm]
            elif mode == "reversed":
                q = (1.0 - q).clamp_min(1e-8)
            elif mode != "learned":
                raise ValueError(f"Unsupported dom_weight_mode={mode!r}")

        q = q.clamp_min(1e-8)
        n_eff = float((q.sum() ** 2) / q.pow(2).sum().clamp_min(1e-12))
        pi = q / q.sum().clamp_min(1e-8)

        stats = {
            "mode": mode,
            "n": n,
            "n_eff": n_eff,
            "n_eff_ratio": n_eff / max(n, 1),
            "q_mean": float(q.mean()),
            "q_min": float(q.min()),
            "q_max": float(q.max()),
            "kappa_q": float(self.dom_kappa_q),
            "tau_q": float(self.dom_tau_q),
            "feat_used": feat_used,
            "center_valid": bool(center_valid),
            "dom_control_seed": int(self.dom_control_seed),
        }
        if s is not None:
            s_np = s.detach().float().cpu()
            stats["s_std"] = float(s_np.std())
            stats["s_p05"] = float(torch.quantile(s_np, 0.05))
            stats["s_p95"] = float(torch.quantile(s_np, 0.95))
            stats["s_p95_p05"] = float(stats["s_p95"] - stats["s_p05"])
            # Top vs bottom weight pool: mean center similarity gap (analysis only).
            k = max(1, n // 20)
            top_idx = torch.topk(q, k).indices
            bot_idx = torch.topk(-q, k).indices
            stats["s_top_minus_bottom"] = float(s[top_idx].mean() - s[bot_idx].mean())
        if y is not None and s is not None and center_valid:
            # Post-hoc only: never used for HP selection.
            try:
                from pygod.utils import eval_roc_auc

                y_bool = y.bool().cpu() if hasattr(y, "bool") else torch.as_tensor(y).bool()
                # Higher q ≈ more "normal/dominant"; invert for anomaly AUROC of weight.
                stats["q_vs_label_auroc_analysis"] = float(
                    eval_roc_auc(y_bool, (-q).detach().cpu())
                )
                rho = max(1, int(0.05 * n))
                top_q = torch.topk(q, rho).indices.cpu()
                stats["top5pct_weight_anomaly_rate"] = float(y_bool[top_q].float().mean())
            except Exception:
                pass
        return pi, q, stats

    def _freeze_ae_eval(self):
        """AE must stay in eval (no dropout) while building fixed latents / weights."""
        if self.ae is None:
            return
        self.ae.eval()
        for param in self.ae.parameters():
            param.requires_grad_(False)

    def _cache_center_latents(self, data):
        """Compute z,h,r once under AE.eval() for paired free/dom training."""
        self._freeze_ae_eval()
        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        with torch.no_grad():
            z, h, r_final = self._build_z(x, edge_index)
            z = self._normalize_clip(z)
        self._cached_center_latents = {
            "z": z.detach(),
            "h": h.detach(),
            "r": r_final.detach(),
            "edge_index": edge_index,
            "x": x,
        }
        return self._cached_center_latents

    def _train_dom_weighted_fm(
        self,
        data,
        ae_path: str,
        model: FlowMatchingModel,
        ckpt_name: str = "dm_dom.pt",
        z_cached: Optional[torch.Tensor] = None,
        pi_cached: Optional[torch.Tensor] = None,
        stats_cached: Optional[dict] = None,
    ):
        """Train dominant-mode flow with a FIXED π (built once before training)."""
        ckpt = os.path.join(ae_path, ckpt_name)
        if os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1" and os.path.exists(ckpt):
            if self.verbose:
                print(f"Reusing FM checkpoint {ckpt_name}")
            return
        if self.verbose:
            print(
                f"Training FM (dom/{self.dom_weight_mode}) -> {ckpt_name} "
                f"(κ={self.dom_kappa_q}, τ={self.dom_tau_q}) ..."
            )

        self._freeze_ae_eval()
        if z_cached is None or pi_cached is None or stats_cached is None:
            cache = self._cached_center_latents or self._cache_center_latents(data)
            z = cache["z"]
            y = getattr(data, "y", None)
            pi, _q, stats = self._center_consistency_weights(cache["h"], y=y)
        else:
            z = z_cached
            pi = pi_cached
            stats = dict(stats_cached)

        # Re-seed so free→dom RNG pairing is identical across weight modes
        # (shuffled must not consume the global RNG for its permutation).
        dom_fm_seed = int(self.dom_control_seed) + 17
        torch.manual_seed(dom_fm_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(dom_fm_seed)

        if self.verbose:
            print(
                f"  dom weights fixed: feat={stats.get('feat_used')} "
                f"valid={stats.get('center_valid')} "
                f"N_eff/N={stats.get('n_eff_ratio'):.3f} "
                f"s_std={stats.get('s_std', float('nan')):.4g}",
                flush=True,
            )

        fm_lr = self.lr * 0.5
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        best_loss = float("inf")
        patience = 0
        graph_context = torch.zeros(1, z.shape[1], device=z.device)

        for epoch in range(self.diff_epochs):
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue
            loss = flow_matching_loss(
                model.velocity_fn, z, graph_context, reduction="mean", weight=pi
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()
            if self.verbose and epoch % 20 == 0:
                print(
                    f"FM-dom Epoch {epoch:04d} loss={loss.item():.6f} "
                    f"N_eff={stats['n_eff']:.1f}/{stats['n']} "
                    f"({stats['n_eff_ratio']:.2f}N) "
                    f"q∈[{stats['q_min']:.3f},{stats['q_max']:.3f}]"
                )
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "gate_state": self.gate_module.state_dict(),
                        "dom_weight_stats": stats,
                    },
                    ckpt,
                )
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-dom early stopping")
                    break

        self._last_n_eff = float(stats["n_eff"])
        self._last_n_eff_ratio = float(stats["n_eff_ratio"])
        self._last_dom_weight_stats = stats
        if not os.path.exists(ckpt):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "gate_state": self.gate_module.state_dict(),
                    "dom_weight_stats": stats,
                },
                ckpt,
            )

    def _train_uncond_fm(
        self,
        data,
        ae_path: str,
        model: FlowMatchingModel,
        ckpt_name: str,
        latent_kind: str = "full",
        also_train_gate: bool = True,
        z_cached: Optional[torch.Tensor] = None,
        z_ref_cached: Optional[torch.Tensor] = None,
    ):
        """Train an unconditional FM on full or residual-suppressed latents."""
        ckpt = os.path.join(ae_path, ckpt_name)
        if os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1" and os.path.exists(ckpt):
            if self.verbose:
                print(f"Reusing FM checkpoint {ckpt_name}")
            return
        if self.verbose:
            print(f"Training FM ({latent_kind}) -> {ckpt_name} ...")

        fm_lr = self.lr * 0.5
        params = list(model.parameters())
        if also_train_gate:
            params = params + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        best_loss = float("inf")
        patience = 0

        # If latents are pre-cached (center_dom), keep AE frozen and reuse fixed z.
        use_cache = z_cached is not None

        for epoch in range(self.diff_epochs):
            if use_cache:
                z = z_cached if latent_kind == "full" else z_ref_cached
                if z is None:
                    raise ValueError(f"Missing cached latent for latent_kind={latent_kind}")
            else:
                x = data.x.cuda().to(torch.float32)
                edge_index = data.edge_index.cuda()
                z_full, z_ref, _, _ = self._build_z_pair(x, edge_index)
                z = z_full if latent_kind == "full" else z_ref
                z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue
            graph_context = torch.zeros(1, z.shape[1], device=z.device)
            loss = flow_matching_loss(model.velocity_fn, z, graph_context, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()
            if self.verbose and epoch % 20 == 0:
                print(f"FM-{latent_kind} Epoch {epoch:04d} loss={loss.item():.6f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save(
                    {"state_dict": model.state_dict(), "gate_state": self.gate_module.state_dict()},
                    ckpt,
                )
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print(f"FM-{latent_kind} early stopping")
                    break
        if not os.path.exists(ckpt):
            torch.save(
                {"state_dict": model.state_dict(), "gate_state": self.gate_module.state_dict()},
                ckpt,
            )

    def _train_shared_two_heads(self, data, ae_path: str, model: MLPFlowMatchingTwoHead):
        ckpt = os.path.join(ae_path, "dm_shared_two_heads.pt")
        if os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1" and os.path.exists(ckpt):
            if self.verbose:
                print("Reusing shared two-head FM checkpoint")
            return
        if self.verbose:
            print("Training shared-backbone two-head FM ...")
        fm_lr = self.lr * 0.5
        params = list(model.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        best_loss = float("inf")
        patience = 0
        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z_full, z_ref, _, _ = self._build_z_pair(x, edge_index)
            z_full = self._normalize_clip(z_full)
            z_ref = self._normalize_clip(z_ref)
            if torch.isnan(z_full).any() or torch.isnan(z_ref).any():
                continue
            ctx = torch.zeros(1, z_full.shape[1], device=z_full.device)
            # Train each head on its own latent target via a thin wrapper.
            loss_full = flow_matching_loss(
                lambda xt, t, context=None, proto_alpha=None: model(xt, t, which="full"),
                z_full,
                ctx,
                reduction="mean",
            )
            loss_ref = flow_matching_loss(
                lambda xt, t, context=None, proto_alpha=None: model(xt, t, which="ref"),
                z_ref,
                ctx,
                reduction="mean",
            )
            loss = loss_full + loss_ref
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()
            if self.verbose and epoch % 20 == 0:
                print(
                    f"FM-shared Epoch {epoch:04d} loss={loss.item():.6f} "
                    f"(full={loss_full.item():.4f}, ref={loss_ref.item():.4f})"
                )
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save(
                    {"state_dict": model.state_dict(), "gate_state": self.gate_module.state_dict()},
                    ckpt,
                )
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-shared early stopping")
                    break
        if not os.path.exists(ckpt):
            torch.save(
                {"state_dict": model.state_dict(), "gate_state": self.gate_module.state_dict()},
                ckpt,
            )

    def _run_dual_flow_trial(self, data, ae_path: str, z_dim: int):
        """One trial under dual_flow_mode controls (no prototype branch)."""
        mode = self.dual_flow_mode
        hid = int(self.flow_hidden_dim)
        wider_hid = max(hid, int(round(hid * (2.0 ** 0.5))))  # ~param-matched to two nets

        self.dm_ref = None
        self.dm_shared = None
        self.dm_proto = None
        self.proto = None

        if mode == "shared_two_heads":
            self.dm_shared = MLPFlowMatchingTwoHead(d_in=z_dim, dim_t=hid).cuda()
            self._train_shared_two_heads(data, ae_path, self.dm_shared)
            state = torch.load(os.path.join(ae_path, "dm_shared_two_heads.pt"))
            self.dm_shared.load_state_dict(state["state_dict"])
            if "gate_state" in state:
                self.gate_module.load_state_dict(state["gate_state"])
            self.dm = None
            return self.sample_dual(data, shared=True)

        dim_t = wider_hid if mode == "wider_single" else hid
        velocity_full = MLPFlowMatching(d_in=z_dim, dim_t=dim_t, cond_dim=None).cuda()
        self.dm = FlowMatchingModel(velocity_fn=velocity_full, hid_dim=z_dim).cuda()

        if mode in ("single_full", "wider_single"):
            self._train_uncond_fm(data, ae_path, self.dm, "dm_full.pt", latent_kind="full")
            state = torch.load(os.path.join(ae_path, "dm_full.pt"))
            self.dm.load_state_dict(state["state_dict"])
            if "gate_state" in state:
                self.gate_module.load_state_dict(state["gate_state"])
            return self.sample(None, self.dm, data)

        if mode == "single_ref":
            self._train_uncond_fm(data, ae_path, self.dm, "dm_ref.pt", latent_kind="ref")
            state = torch.load(os.path.join(ae_path, "dm_ref.pt"))
            self.dm.load_state_dict(state["state_dict"])
            if "gate_state" in state:
                self.gate_module.load_state_dict(state["gate_state"])
            return self.sample(None, self.dm, data)

        # center_dom: free (uniform) + dominant (fixed center weights) on full latent
        if mode == "center_dom":
            self._freeze_ae_eval()
            cache = self._cache_center_latents(data)
            z = cache["z"]
            y = getattr(data, "y", None)
            with torch.no_grad():
                pi, q, stats = self._center_consistency_weights(cache["h"], y=y)
            self._train_uncond_fm(
                data, ae_path, self.dm, "dm_free.pt", latent_kind="full", z_cached=z
            )
            state = torch.load(os.path.join(ae_path, "dm_free.pt"))
            self.dm.load_state_dict(state["state_dict"])
            if "gate_state" in state:
                self.gate_module.load_state_dict(state["gate_state"])
            velocity_dom = MLPFlowMatching(d_in=z_dim, dim_t=hid, cond_dim=None).cuda()
            self.dm_ref = FlowMatchingModel(velocity_fn=velocity_dom, hid_dim=z_dim).cuda()
            self._train_dom_weighted_fm(
                data,
                ae_path,
                self.dm_ref,
                "dm_dom.pt",
                z_cached=z,
                pi_cached=pi,
                stats_cached=stats,
            )
            state_dom = torch.load(os.path.join(ae_path, "dm_dom.pt"))
            self.dm_ref.load_state_dict(state_dom["state_dict"])
            if state_dom.get("dom_weight_stats"):
                st = state_dom["dom_weight_stats"]
                self._last_n_eff = float(st.get("n_eff", float("nan")))
                self._last_n_eff_ratio = float(st.get("n_eff_ratio", float("nan")))
                self._last_dom_weight_stats = st
            return self.sample_dual(data, shared=False)

        # Dual modes: full + reference nets
        self._train_uncond_fm(data, ae_path, self.dm, "dm_full.pt", latent_kind="full")
        state = torch.load(os.path.join(ae_path, "dm_full.pt"))
        self.dm.load_state_dict(state["state_dict"])
        if "gate_state" in state:
            self.gate_module.load_state_dict(state["gate_state"])

        ref_kind = "full" if mode == "duplicate_dual" else "ref"
        velocity_ref = MLPFlowMatching(d_in=z_dim, dim_t=hid, cond_dim=None).cuda()
        self.dm_ref = FlowMatchingModel(velocity_fn=velocity_ref, hid_dim=z_dim).cuda()
        ref_ckpt = "dm_full_b.pt" if mode == "duplicate_dual" else "dm_ref.pt"
        self._train_uncond_fm(
            data, ae_path, self.dm_ref, ref_ckpt, latent_kind=ref_kind, also_train_gate=False
        )
        state_ref = torch.load(os.path.join(ae_path, ref_ckpt))
        self.dm_ref.load_state_dict(state_ref["state_dict"])
        return self.sample_dual(data, shared=False)

    def sample_dual(self, data, shared: bool = False):
        """Inference for dual-flow guidance: v=(1+w)v_full - w v_ref."""
        self.ae.eval()
        if shared:
            self.dm_shared.eval()
        else:
            self.dm.eval()
            self.dm_ref.eval()

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        z_data, _, _ = self._build_z(x, edge_index)
        z0 = self._normalize_clip(z_data)
        inference_seed = int(os.environ.get("FMGAD_INFERENCE_SEED", "1729"))
        noise_gen = torch.Generator(device=z0.device)
        noise_gen.manual_seed(inference_seed)
        noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=noise_gen)
        s = to_dense_adj(edge_index)[0].cuda()
        num_steps = 1
        reconstructed = sample_flow_matching_pair(
            self.dm_shared if shared else self.dm.velocity_fn,
            None if shared else self.dm_ref.velocity_fn,
            noise,
            num_steps=num_steps,
            weight=self.weight,
            shared_two_head=shared,
        )
        h_hat = reconstructed[:, : self.hid_dim]
        x_, s_ = self.ae.decode(h_hat, edge_index)
        score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
        if edge_index.numel() > 0:
            score = smooth_scores_by_graph(score, edge_index, self.score_smoothing_alpha, score.device)
        if not bool(getattr(self, "ensemble_score", False)):
            score = self._apply_score_polarity_adapter(score)

        y_eval = data.y.bool()
        scores_cpu = score.detach().cpu()
        if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
            scores_cpu = torch.nan_to_num(scores_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        pyg_auc = eval_roc_auc(y_eval, scores_cpu)
        pyg_ap = eval_average_precision(y_eval, scores_cpu)
        pyg_rec = eval_recall_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        pyg_prec = eval_precision_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        p, r, _ = precision_recall_curve(y_eval.numpy(), scores_cpu.numpy())
        pyg_auprc = auc(r, p)
        f1_at_k = (
            2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec) if (pyg_prec + pyg_rec) > 0 else 0.0
        )
        if self.verbose:
            print(
                "dual_mode:{},steps:{},pyg_AUC: {:.4f}, pyg_AP: {:.4f}".format(
                    self.dual_flow_mode, num_steps, pyg_auc, pyg_ap
                )
            )
        if getattr(self, "ensemble_score", False):
            return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k), scores_cpu.clone()
        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k)

    def forward(self, dset: str):
        def _sync_cuda() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.dataset = dset
        data = self._load_dataset(dset)
        num_nodes = int(getattr(data, "num_nodes", data.x.size(0)))
        num_edges = int(data.edge_index.size(1)) if hasattr(data, "edge_index") else 0
        num_features = int(data.x.size(1)) if hasattr(data, "x") and data.x.dim() == 2 else 0
        train_time_sec = None
        inference_time_sec = None
        peak_gpu_mem_mb = None
        peak_gpu_reserved_mb = None
        num_parameters = None
        if bool(getattr(self, "profile_efficiency", False)):
            if torch.cuda.is_available():
                _sync_cuda()
                torch.cuda.reset_peak_memory_stats()
            train_time_sec = 0.0
            inference_time_sec = 0.0
        self._last_polarity_diag = None
        if self.polarity_enabled:
            if self.verbose:
                print("Precomputing local_prior polarity probe...", flush=True)
            self._local_prior_probe = compute_local_prior(data.x.cpu(), data.edge_index.cpu())
        else:
            self._local_prior_probe = None
        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)

        # AE
        self.ae = GraphAE(in_dim=data.num_node_features, hid_dim=self.hid_dim, dropout=self.ae_dropout).cuda()
        save_dir = self._ensure_save_dir(dset)
        ae_path = os.path.join(
            save_dir,
            f"ae_drop{self.ae_dropout}_lr{self.ae_lr}_alpha{self.ae_alpha}_hid{self.hid_dim}",
        )
        # Avoid multiple processes writing the same checkpoint path when sharing an exp_tag.
        run_tag = self.exp_tag if self.exp_tag else f"run_{os.getpid()}_{int(time.time() * 1000)}"
        _tag_suffix = os.environ.get("FMGAD_RUN_TAG_SUFFIX", "").strip()
        if _tag_suffix:
            run_tag = f"{run_tag}_{_tag_suffix}"
        ae_path = os.path.join(ae_path, run_tag)
        os.makedirs(ae_path, exist_ok=True)

        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            _t_train = time.perf_counter()
        ae_ckpt = self._train_ae_once(data, ae_path)
        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            train_time_sec += time.perf_counter() - _t_train
        if self.verbose:
            print(f"loading AE checkpoint: {ae_ckpt:04d}")
        ae_dict = torch.load(os.path.join(ae_path, f"{ae_ckpt}.pt"))
        self.ae.load_state_dict(ae_dict["state_dict"])
        # Freeze AE: dropout must be off when building fixed latents / center weights.
        self._freeze_ae_eval()
        self.gate_module = self.gate_module.to(next(self.ae.parameters()).device)

        # 2) trials
        num_trial = getattr(self, "num_trial", 3)
        dm_auc, dm_ap, dm_rec, dm_auprc, dm_f1 = [], [], [], [], []

        for _ in tqdm.tqdm(range(num_trial)):
            # z_dim = 2*hid_dim
            z_dim = 2 * self.hid_dim

            if self.dual_flow_mode is not None:
                if self.generative_backend != "flow":
                    raise ValueError("dual_flow_mode requires generative_backend=flow")
                ret = self._run_dual_flow_trial(data, ae_path, z_dim)
            elif self.generative_backend == "diffusion":
                denoise_free = MLPDiffusion(d_in=z_dim, dim_t=512).cuda()
                self.dm = DiffusionModel(denoise_fn=denoise_free, hid_dim=z_dim).cuda()
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    _t_train = time.perf_counter()
                proto_h = self._train_dm_free(data, ae_path)
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    train_time_sec += time.perf_counter() - _t_train

                dm_dict = torch.load(os.path.join(ae_path, "dm_self.pt"))
                self.dm.load_state_dict(dm_dict["state_dict"])
                if "gate_state" in dm_dict:
                    self.gate_module.load_state_dict(dm_dict["gate_state"])
                self.proto = dm_dict.get("prototype")
                self.dm_proto = None
                self.proto = None
                ret = self.sample(None, self.dm, data)
            else:
                # Free-flow model: cond_dim=None => zero context vector.
                velocity_free = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=None).cuda()
                self.dm = FlowMatchingModel(velocity_fn=velocity_free, hid_dim=z_dim).cuda()
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    _t_train = time.perf_counter()
                proto_h = self._train_dm_free(data, ae_path)
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    train_time_sec += time.perf_counter() - _t_train

                dm_dict = torch.load(os.path.join(ae_path, "dm_self.pt"))
                self.dm.load_state_dict(dm_dict["state_dict"])
                if "gate_state" in dm_dict:
                    self.gate_module.load_state_dict(dm_dict["gate_state"])
                self.proto = dm_dict.get("prototype")

                if bool(getattr(self, "use_proto", True)):
                    # Proto model: cond_dim = hid_dim (condition on prototype in h-space).
                    velocity_proto = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=self.hid_dim).cuda()
                    self.dm_proto = FlowMatchingModel(velocity_fn=velocity_proto, hid_dim=z_dim).cuda()
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        _t_train = time.perf_counter()
                    self._train_dm_proto(data, ae_path)
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        train_time_sec += time.perf_counter() - _t_train
                    dm_proto_dict = torch.load(os.path.join(ae_path, "proto_dm_self.pt"))
                    self.dm_proto.load_state_dict(dm_proto_dict["state_dict"])
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        _t_infer = time.perf_counter()
                    ret = self.sample(self.dm_proto, self.dm, data)
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        inference_time_sec += time.perf_counter() - _t_infer
                else:
                    # Hard no-proto: no prototype-conditioned branch in training/inference.
                    self.dm_proto = None
                    self.proto = None
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        _t_infer = time.perf_counter()
                    ret = self.sample(None, self.dm, data)
                    if bool(getattr(self, "profile_efficiency", False)):
                        _sync_cuda()
                        inference_time_sec += time.perf_counter() - _t_infer
            if len(ret) == 6:
                auc_this, ap_this, rec_this, auprc_this, f1_this, scores = ret
                if not hasattr(self, "_ensemble_scores"):
                    self._ensemble_scores = []
                self._ensemble_scores.append(scores)
            else:
                auc_this, ap_this, rec_this, auprc_this, f1_this = ret
            dm_auc.append(auc_this)
            dm_ap.append(ap_this)
            dm_rec.append(rec_this)
            dm_auprc.append(auprc_this)
            dm_f1.append(f1_this)

        if getattr(self, "ensemble_score", False) and hasattr(self, "_ensemble_scores") and len(self._ensemble_scores) > 0:
            # Average scores across trials, then compute metrics once on the mean score.
            stacked = torch.stack(self._ensemble_scores)  # [num_trial, N]
            mean_scores = stacked.mean(dim=0)  # [N]
            if torch.isnan(mean_scores).any() or torch.isinf(mean_scores).any():
                mean_scores = torch.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)
            mean_scores = self._apply_score_polarity_adapter(mean_scores)

            y_eval = data.y  # evaluation labels only (after all score / polarity steps)

            pyg_auc = eval_roc_auc(y_eval, mean_scores)
            pyg_ap = eval_average_precision(y_eval, mean_scores)
            pyg_rec = eval_recall_at_k(y_eval, mean_scores, int(y_eval.sum()))
            pyg_prec = eval_precision_at_k(y_eval, mean_scores, int(y_eval.sum()))

            y_np = y_eval.cpu().numpy()
            p, r, _ = precision_recall_curve(y_np, mean_scores.cpu().numpy())
            pyg_auprc = auc(r, p)
            pyg_f1 = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec) if (pyg_prec + pyg_rec) > 0 else 0.0
            dm_auc = torch.tensor([float(pyg_auc)])
            dm_ap = torch.tensor([float(pyg_ap)])
            dm_rec = torch.tensor([float(pyg_rec)])
            dm_auprc = torch.tensor([float(pyg_auprc)])
            dm_f1 = torch.tensor([float(pyg_f1)])
            # Keep for optional unsupervised model selection (e.g. AutoGAD CST).
            self._last_scores = mean_scores.detach().cpu()
            del self._ensemble_scores
        else:
            dm_auc = torch.tensor(dm_auc)
            dm_ap = torch.tensor(dm_ap)
            dm_rec = torch.tensor(dm_rec)
            dm_auprc = torch.tensor(dm_auprc)
            dm_f1 = torch.tensor(dm_f1)

        print(
            "Final AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Final Recall: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AUPRC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final F1@k: {:.4f}±{:.4f} ({:.4f})".format(
                torch.mean(dm_auc),
                torch.std(dm_auc),
                torch.max(dm_auc),
                torch.mean(dm_ap),
                torch.std(dm_ap),
                torch.max(dm_ap),
                torch.mean(dm_rec),
                torch.std(dm_rec),
                torch.max(dm_rec),
                torch.mean(dm_auprc),
                torch.std(dm_auprc),
                torch.max(dm_auprc),
                torch.mean(dm_f1),
                torch.std(dm_f1),
                torch.max(dm_f1),
            )
        )

        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            if torch.cuda.is_available():
                peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                peak_gpu_reserved_mb = float(torch.cuda.max_memory_reserved() / 1024.0 / 1024.0)
            param_count = 0
            modules_for_count = [self.ae, self.dm, self.gate_module]
            if getattr(self, "dm_proto", None) is not None:
                modules_for_count.append(self.dm_proto)
            for mod in modules_for_count:
                if mod is not None:
                    param_count += int(sum(p.numel() for p in mod.parameters()))
            num_parameters = int(param_count)

        out = {
            "auc_mean": float(torch.mean(dm_auc)),
            "auc_std": float(torch.std(dm_auc)),
            "ap_mean": float(torch.mean(dm_ap)),
            "ap_std": float(torch.std(dm_ap)),
            "rec_mean": float(torch.mean(dm_rec)),
            "rec_std": float(torch.std(dm_rec)),
            "auprc_mean": float(torch.mean(dm_auprc)),
            "auprc_std": float(torch.std(dm_auprc)),
            "f1_mean": float(torch.mean(dm_f1)),
            "f1_std": float(torch.std(dm_f1)),
            "polarity_enabled": self.polarity_enabled,
            "polarity_diagnostics": self._last_polarity_diag,
            "generative_backend": self.generative_backend,
            "use_proto": bool(self.use_proto),
            "dual_flow_mode": self.dual_flow_mode,
            "dom_weight_mode": getattr(self, "dom_weight_mode", None),
            "n_eff": float(getattr(self, "_last_n_eff", float("nan"))),
            "n_eff_ratio": float(getattr(self, "_last_n_eff_ratio", float("nan"))),
            "dom_weight_stats": getattr(self, "_last_dom_weight_stats", None),
            "guidance_gap": float(getattr(self, "_last_guidance_gap", float("nan"))),
            "proto_mode": str(getattr(self, "_last_proto_mode", "learned")),
            "proto_m": int(os.environ.get("FMGAD_PROTO_M", "1")),
            "weight": float(self.weight),
        }
        if self.proto is not None:
            p = self.proto.detach().float().cpu().reshape(-1)
            out["prototype"] = [float(x) for x in p.tolist()]
        if os.environ.get("FMGAD_SAVE_SCORES", "0") == "1" and getattr(self, "_last_scores", None) is not None:
            out["scores"] = [float(x) for x in self._last_scores.reshape(-1).tolist()]
        if bool(getattr(self, "profile_efficiency", False)):
            out.update(
                {
                    "profile_efficiency": True,
                    "train_time_sec": float(train_time_sec) if train_time_sec is not None else None,
                    "inference_time_sec": float(inference_time_sec) if inference_time_sec is not None else None,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                    "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                    "num_parameters": num_parameters,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "num_features": num_features,
                    "sample_steps": int(self.sample_steps),
                }
            )
        return out

    def _train_ae_once(self, data, ae_path: str) -> int:
        if os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1":
            epochs = [
                int(name[:-3])
                for name in os.listdir(ae_path)
                if name.endswith(".pt") and name[:-3].isdigit()
            ]
            if epochs:
                best_epoch = max(epochs)
                if self.verbose:
                    print(f"Reusing AE checkpoint: {best_epoch:04d}")
                return best_epoch
        if self.verbose:
            print("Training autoencoder...")

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.ae_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        best_epoch = 0
        patience = 0

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        s = to_dense_adj(edge_index)[0].cuda()

        for epoch in range(1, self.ae_epochs + 1):
            self.ae.train()
            optimizer.zero_grad()

            x_, s_, _ = self.ae(x, edge_index)
            score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
            loss = torch.mean(score)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                patience = 0
                torch.save({"state_dict": self.ae.state_dict()}, os.path.join(ae_path, f"{best_epoch}.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("AE early stopping")
                    break

            if self.verbose and epoch % 50 == 0:
                print(f"AE Epoch {epoch:04d} loss={loss.item():.6f}")

        return best_epoch

    def _normalize_clip(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _, _ = self._normalize_clip_with_stats(inputs)
        return x

    def _normalize_clip_with_stats(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = inputs.mean(dim=0, keepdim=True)
        std = inputs.std(dim=0, keepdim=True) + 1e-8
        x = (inputs - mean) / std
        x = torch.clamp(x, -10.0, 10.0)
        return x, mean, std

    def _denormalize(self, normed: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return normed * std + mean

    def _train_dm_free(self, data, ae_path: str) -> torch.Tensor:
        dm_path = os.path.join(ae_path, "dm_self.pt")
        if os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1" and os.path.exists(dm_path):
            state = torch.load(dm_path, map_location="cpu")
            if self.verbose:
                print(f"Reusing {self.generative_backend} free checkpoint")
            return state.get("prototype")

        if self.generative_backend == "diffusion":
            return self._train_diffusion_free(data, ae_path)

        if self.verbose:
            print("Training FM free model...")

        fm_lr = self.lr * 0.5
        params = list(self.dm.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0
        proto_h = None
        with torch.no_grad():
            x0 = data.x.cuda().to(torch.float32)
            e0 = data.edge_index.cuda()
            _, h0, _ = self._build_z(x0, e0)
            proto_h_init = torch.mean(h0, dim=0).detach()

        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z, h, r_final = self._build_z(x, edge_index)

            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            graph_context = torch.zeros(1, z.shape[1], device=z.device)
            loss = flow_matching_loss(self.dm.velocity_fn, z, graph_context, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            with torch.no_grad():
                noise = torch.randn_like(z)
                reconstructed = sample_flow_matching(self.dm.velocity_fn, noise, num_steps=10, proto=None, proto_alpha=None)
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    reconstructed = z.clone()
                recon_h = reconstructed[:, : self.hid_dim]

            if epoch == 0:
                proto_h = torch.mean(h, dim=0)  # [hid_dim]
            else:
                proto_expanded = proto_h.unsqueeze(0)
                s_v = self.cos(proto_expanded, recon_h)
                weight = softmax_with_temperature(s_v, t=5).reshape(1, -1)
                proto_h = torch.mm(weight, recon_h).squeeze(0).detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"FM-free Epoch {epoch:04d} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                save_dict = {
                    "state_dict": self.dm.state_dict(),
                    "prototype": proto_h,
                    "gate_state": self.gate_module.state_dict(),
                }
                torch.save(save_dict, os.path.join(ae_path, "dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-free early stopping")
                    break

        if not os.path.exists(dm_path):
            proto_fallback = proto_h if proto_h is not None else proto_h_init
            save_dict = {
                "state_dict": self.dm.state_dict(),
                "prototype": proto_fallback,
                "gate_state": self.gate_module.state_dict(),
            }
            torch.save(save_dict, dm_path)
            if self.verbose:
                print("FM-free: fallback save")

        return proto_h

    def _train_diffusion_free(self, data, ae_path: str) -> Optional[torch.Tensor]:
        """Train free-only EDM on residual-augmented z (no prototype guidance)."""
        dm_path = os.path.join(ae_path, "dm_self.pt")
        if self.verbose:
            print("Training Diffusion free model (EDM, residual-z)...")

        dm_lr = self.lr * 0.5
        params = list(self.dm.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params, lr=dm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0
        proto_h = None
        with torch.no_grad():
            x0 = data.x.cuda().to(torch.float32)
            e0 = data.edge_index.cuda()
            _, h0, _ = self._build_z(x0, e0)
            proto_h_init = torch.mean(h0, dim=0).detach()

        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z, h, _ = self._build_z(x, edge_index)
            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            loss, _, reconstructed = self.dm(z)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            if epoch == 0:
                proto_h = torch.mean(h, dim=0)
            else:
                recon_h = reconstructed[:, : self.hid_dim].detach()
                if torch.isnan(recon_h).any() or torch.isinf(recon_h).any():
                    recon_h = h.detach()
                proto_expanded = proto_h.unsqueeze(0)
                s_v = self.cos(proto_expanded, recon_h)
                weight = softmax_with_temperature(s_v, t=5).reshape(1, -1)
                proto_h = torch.mm(weight, recon_h).squeeze(0).detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"Diff-free Epoch {epoch:04d} loss={float(loss):.6f}")

            if float(loss) < best_loss:
                best_loss = float(loss)
                patience = 0
                torch.save(
                    {
                        "state_dict": self.dm.state_dict(),
                        "prototype": proto_h,
                        "gate_state": self.gate_module.state_dict(),
                        "generative_backend": "diffusion",
                    },
                    dm_path,
                )
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("Diff-free early stopping")
                    break

        if not os.path.exists(dm_path):
            torch.save(
                {
                    "state_dict": self.dm.state_dict(),
                    "prototype": proto_h if proto_h is not None else proto_h_init,
                    "gate_state": self.gate_module.state_dict(),
                    "generative_backend": "diffusion",
                },
                dm_path,
            )
            if self.verbose:
                print("Diff-free: fallback save")
        return proto_h

    def _train_dm_proto(self, data, ae_path: str):
        proto_path = os.path.join(ae_path, "proto_dm_self.pt")
        n_proto = max(int(os.environ.get("FMGAD_PROTO_M", "1")), 1)
        # Multi-proto needs its own proto checkpoint; never reuse M=1 weights.
        reuse_ok = (
            os.environ.get("FMGAD_REUSE_CHECKPOINTS", "0") == "1"
            and n_proto == 1
            and os.path.exists(proto_path)
        )
        if reuse_ok:
            if self.verbose:
                print("Reusing FM proto checkpoint")
            return
        if self.verbose:
            print(f"Training FM proto model... (M={n_proto})")
        if n_proto > 1 and os.path.exists(proto_path):
            # Force overwrite of single-proto weights.
            try:
                os.remove(proto_path)
            except OSError:
                pass

        fm_lr = self.lr * 0.5
        params_proto = list(self.dm_proto.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params_proto, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0

        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z, h, r_final = self._build_z(x, edge_index)
            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            n_proto = max(int(os.environ.get("FMGAD_PROTO_M", "1")), 1)
            if n_proto > 1:
                with torch.no_grad():
                    feats = r_final.detach()
                    if torch.allclose(feats.std(), torch.zeros((), device=feats.device), atol=1e-8):
                        feats = z.detach()
                    centers, soft = self._soft_kmeans(feats, n_proto)
                    if centers.shape[1] == self.hid_dim:
                        proto_context = soft @ centers
                    else:
                        proto_context = soft @ centers[:, : self.hid_dim]
            else:
                proto_context = self.proto.unsqueeze(0) if self.proto.dim() == 1 else self.proto.mean(dim=0, keepdim=True)

            sample_weight = None
            if (
                n_proto == 1
                and bool(getattr(self, "use_proto_normal_weight", True))
                and self.proto is not None
            ):
                # L_normal with q_i = sg[φ(sim_i)]. AE embeddings h often collapse
                # (constant across nodes), so score normality in residual-augmented
                # z-space: c_z = [c_h; mean(r)], φ = softmax(τ · cos(z, c_z)).
                with torch.no_grad():
                    c_h = proto_context.reshape(1, -1)
                    if r_final.shape[1] == c_h.shape[1]:
                        c_z = torch.cat([c_h, r_final.mean(dim=0, keepdim=True)], dim=1)
                        cos_sim = torch.nn.functional.cosine_similarity(
                            z, c_z.expand(z.shape[0], -1), dim=1
                        )
                    else:
                        cos_sim = self.cos(h, c_h.expand(h.shape[0], -1))
                    tau = float(getattr(self, "proto_normal_temp", 50.0))
                    sample_weight = torch.softmax(tau * cos_sim, dim=0)
                    if float(sample_weight.sum()) < 1e-8:
                        sample_weight = torch.ones_like(sample_weight) / float(sample_weight.numel())
            loss = conditional_flow_matching_loss(
                self.dm_proto.velocity_fn,
                z,
                proto_context,
                t_sampling="logit_normal",
                reduction="mean",
                weight=sample_weight,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_proto, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                wmsg = ""
                if sample_weight is not None:
                    wmsg = (
                        f" q_mean={float(sample_weight.mean()):.4f}"
                        f" q_max={float(sample_weight.max()):.4f}"
                        f" q_min={float(sample_weight.min()):.4f}"
                        f" q_ratio={float(sample_weight.max() / (sample_weight.min() + 1e-12)):.2f}"
                    )
                print(f"FM-proto Epoch {epoch:04d} loss={loss.item():.6f}{wmsg}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save({"state_dict": self.dm_proto.state_dict()}, os.path.join(ae_path, "proto_dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-proto early stopping")
                    break

        if not os.path.exists(proto_path):
            torch.save({"state_dict": self.dm_proto.state_dict()}, proto_path)
            if self.verbose:
                print("FM-proto: fallback save")

    def _resolve_proto_context(self, data, z: torch.Tensor, h: torch.Tensor, r_final: torch.Tensor):
        """Build proto context for inference; supports diagnostic overrides.

        Env:
          FMGAD_PROTO_MODE: learned|shuffle|random|zero|none  (default learned)
          FMGAD_PROTO_M: int >=1 multi-prototype soft assignment (default 1)
        """
        mode = os.environ.get("FMGAD_PROTO_MODE", "learned").strip().lower()
        n_proto = max(int(os.environ.get("FMGAD_PROTO_M", "1")), 1)
        device = z.device
        n = z.shape[0]
        hid = int(self.hid_dim)

        if mode in ("none", "off", "no", "noproto"):
            return None, mode, n_proto

        if n_proto > 1:
            # Cluster residual features (h often collapses to a constant).
            feats = r_final.detach()
            if torch.allclose(feats.std(), torch.zeros((), device=device), atol=1e-8):
                feats = z.detach()
            centers, soft = self._soft_kmeans(feats, n_proto)
            # Map centers back to hid_dim context: use residual centers directly
            # when feat dim == hid, else first hid dims of z-centers.
            if centers.shape[1] == hid:
                ctx = soft @ centers  # [N, hid]
            else:
                ctx = soft @ centers[:, :hid]
            return ctx, f"{mode}_m{n_proto}", n_proto

        # Single global prototype branch
        base = None
        if self.proto is not None:
            base = self.proto.detach()
            if base.dim() > 1:
                base = base.mean(dim=0)
            base = base.reshape(-1).to(device)
            if base.numel() != hid:
                # Fallback: mean h
                base = h.mean(dim=0).detach()

        if mode == "learned":
            if base is None:
                return None, mode, n_proto
            return base.unsqueeze(0), mode, n_proto
        if mode == "shuffle":
            # Permute node embeddings as a fake "prototype" source, then mean
            # — destroys semantic alignment while keeping magnitude statistics.
            if base is None:
                return None, mode, n_proto
            perm = torch.randperm(n, device=device)
            shuffled = h[perm].mean(dim=0)
            # Keep same norm as learned proto if possible
            if float(base.norm()) > 1e-8:
                shuffled = shuffled / (shuffled.norm() + 1e-8) * base.norm()
            return shuffled.unsqueeze(0), mode, n_proto
        if mode == "random":
            rnd = torch.randn(hid, device=device, dtype=z.dtype)
            if base is not None and float(base.norm()) > 1e-8:
                rnd = rnd / (rnd.norm() + 1e-8) * base.norm()
            return rnd.unsqueeze(0), mode, n_proto
        if mode == "zero":
            return torch.zeros(1, hid, device=device, dtype=z.dtype), mode, n_proto
        # default
        if base is None:
            return None, mode, n_proto
        return base.unsqueeze(0), mode, n_proto

    @staticmethod
    def _soft_kmeans(feats: torch.Tensor, m: int, iters: int = 20, temp: float = 1.0):
        """Lightweight soft k-means; returns centers [M,D] and soft assign [N,M]."""
        n, d = feats.shape
        m = min(m, n)
        # init: random subset
        idx = torch.randperm(n, device=feats.device)[:m]
        centers = feats[idx].clone()
        for _ in range(iters):
            dist2 = torch.cdist(feats, centers, p=2).pow(2)  # [N,M]
            soft = torch.softmax(-dist2 / max(temp, 1e-6), dim=1)
            dens = soft.sum(dim=0).clamp_min(1e-6).unsqueeze(1)
            centers = (soft.t() @ feats) / dens
        dist2 = torch.cdist(feats, centers, p=2).pow(2)
        soft = torch.softmax(-dist2 / max(temp, 1e-6), dim=1)
        return centers, soft

    def sample(self, proto_model, free_model, data):
        self.ae.eval()
        if proto_model is not None:
            proto_model.eval()
        free_model.eval()
        self._last_guidance_gap = float("nan")
        self._last_proto_mode = os.environ.get("FMGAD_PROTO_MODE", "learned")

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()

        z_data, h, r_final = self._build_z(x, edge_index)
        z0 = self._normalize_clip(z_data)
        # Decouple inference randomness from training duration / early stopping.
        # The same trained checkpoint must produce the same score regardless
        # of how many random numbers were consumed during training.
        inference_seed = int(os.environ.get("FMGAD_INFERENCE_SEED", "1729"))
        noise_gen = torch.Generator(device=z0.device)
        noise_gen.manual_seed(inference_seed)
        noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=noise_gen)

        s = to_dense_adj(edge_index)[0].cuda()

        if self.generative_backend == "diffusion":
            # Free-only EDM sampling from noise; AE reconstruction error for scoring.
            num_steps = max(int(self.sample_steps), 1)
            reconstructed = sample_dm(free_model.denoise_fn_D, noise, num_steps=num_steps)
        else:
            proto_net = proto_model.velocity_fn if proto_model is not None else None
            free_net = free_model.velocity_fn
            proto_context, mode_tag, n_proto = self._resolve_proto_context(data, z0, h, r_final)
            self._last_proto_mode = mode_tag

            # Fix step to 1 and avoid label-based step selection for FM.
            num_steps = 1
            use_proto_branch = (
                bool(getattr(self, "use_proto", True))
                and proto_net is not None
                and proto_context is not None
            )
            if use_proto_branch:
                out = sample_flow_matching_free(
                    proto_net,
                    free_net,
                    noise,
                    num_steps,
                    proto=proto_context,
                    proto_alpha=self.proto_alpha,
                    weight=self.weight,
                    return_gap=True,
                )
                reconstructed, gap = out
                self._last_guidance_gap = float(gap)
            else:
                # Hard no-proto: free-only velocity branch for inference.
                reconstructed = sample_flow_matching(
                    free_net,
                    noise,
                    num_steps=num_steps,
                    proto=None,
                    proto_alpha=None,
                )
                self._last_guidance_gap = 0.0

        h_hat = reconstructed[:, : self.hid_dim]
        x_, s_ = self.ae.decode(h_hat, edge_index)
        score_recon = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

        raw_score = score_recon
        score = raw_score

        # Graph score smoothing is permanently enabled.
        if edge_index.numel() > 0:
            score = smooth_scores_by_graph(score, edge_index, self.score_smoothing_alpha, score.device)

        if not bool(getattr(self, "ensemble_score", False)):
            score = self._apply_score_polarity_adapter(score)

        # Ground-truth labels: evaluation only (must not affect score path above).
        y_eval = data.y.bool()

        scores_cpu = score.detach().cpu()
        if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
            scores_cpu = torch.nan_to_num(scores_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        pyg_auc = eval_roc_auc(y_eval, scores_cpu)
        pyg_ap = eval_average_precision(y_eval, scores_cpu)
        pyg_rec = eval_recall_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        pyg_prec = eval_precision_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        p, r, _ = precision_recall_curve(y_eval.numpy(), scores_cpu.numpy())
        pyg_auprc = auc(r, p)

        if (pyg_prec + pyg_rec) > 0:
            f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
        else:
            f1_at_k = 0.0

        if self.verbose:
            print(
                "steps:{},pyg_AUC: {:.4f}, pyg_AP: {:.4f}, pyg_Recall: {:.4f}, F1@k: {:.4f}, AUPRC: {:.4f}".format(
                    num_steps, pyg_auc, pyg_ap, pyg_rec, f1_at_k, pyg_auprc
                )
            )

        if getattr(self, "ensemble_score", False):
            return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k), scores_cpu.clone()
        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k)
