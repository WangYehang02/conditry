#!/usr/bin/env python3
"""Diagnose AE embedding collapse vs cosine-metric failure (books / disney).

Analysis only — trains/loads AE, does NOT train flow.
Three checks from the protocol:
  1) effective rank of centered H
  2) direction vs norm (std cos, std ||h||, mean pairwise cos)
  3) standardized Euclidean distance to center + post-hoc AUROC vs labels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fmgad.cli import _set_seed
from fmgad.detector import ResFlowGAD
from pygod.utils import eval_roc_auc, load_data

DATASETS = ("books", "disney")
SEEDS = (0, 42)


def _effective_rank(H: torch.Tensor) -> Dict[str, float]:
    # H: [N, d]
    Hc = H - H.mean(dim=0, keepdim=True)
    # economy SVD
    try:
        s = torch.linalg.svdvals(Hc)
    except Exception:
        s = torch.svd(Hc, compute_uv=False).S
    s = s.clamp_min(0.0)
    num = float(s.sum() ** 2)
    den = float(s.pow(2).sum().clamp_min(1e-12))
    r_eff = num / den
    return {
        "r_eff": r_eff,
        "rank_full": int(H.shape[1]),
        "sigma_max": float(s[0]) if s.numel() else float("nan"),
        "sigma_min": float(s[-1]) if s.numel() else float("nan"),
        "sigma_ratio_max_min": float(s[0] / s[-1].clamp_min(1e-12)) if s.numel() else float("nan"),
    }


def _pairwise_mean_cos(H: torch.Tensor, max_pairs: int = 200000) -> float:
    n = H.shape[0]
    Hn = torch.nn.functional.normalize(H, dim=1, eps=1e-8)
    if n * (n - 1) // 2 <= max_pairs:
        g = Hn @ Hn.T
        # exclude diagonal
        off = g[~torch.eye(n, dtype=torch.bool, device=H.device)]
        return float(off.mean())
    # subsample pairs
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    i = torch.randint(0, n, (max_pairs,), generator=gen)
    j = torch.randint(0, n, (max_pairs,), generator=gen)
    mask = i != j
    i, j = i[mask], j[mask]
    return float((Hn[i] * Hn[j]).sum(dim=1).mean())


def _diagnose_H(H: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    H = H.detach().float()
    n, d = H.shape
    c = H.mean(dim=0, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(H, c.expand(n, -1), dim=1)
    norms = torch.norm(H, dim=1)

    # standardized Euclidean distance to center
    mu = H.mean(dim=0)
    sigma = H.std(dim=0, unbiased=False).clamp_min(1e-6)
    d_i = torch.norm((H - mu) / sigma, dim=1)

    out: Dict[str, Any] = {
        "n": int(n),
        "d": int(d),
        "h_std_all": float(H.std()),
        "h_row_std_mean": float(H.std(dim=1).mean()),
        **_effective_rank(H),
        "std_cos_to_center": float(cos.std()),
        "mean_cos_to_center": float(cos.mean()),
        "std_norm": float(norms.std()),
        "mean_norm": float(norms.mean()),
        "cv_norm": float(norms.std() / norms.mean().clamp_min(1e-8)),
        "mean_pairwise_cos": _pairwise_mean_cos(H),
        "std_euclid_center": float(d_i.std()),
        "mean_euclid_center": float(d_i.mean()),
        "euclid_p05": float(torch.quantile(d_i, 0.05)),
        "euclid_p95": float(torch.quantile(d_i, 0.95)),
    }

    y_bool = y.bool().cpu()
    # Higher distance / lower cos ≈ more anomalous for post-hoc analysis only
    try:
        out["auroc_neg_cos_vs_label"] = float(eval_roc_auc(y_bool, (-cos).cpu()))
    except Exception as e:
        out["auroc_neg_cos_vs_label"] = None
        out["auroc_neg_cos_error"] = str(e)
    try:
        out["auroc_euclid_vs_label"] = float(eval_roc_auc(y_bool, d_i.cpu()))
    except Exception as e:
        out["auroc_euclid_vs_label"] = None
        out["auroc_euclid_error"] = str(e)
    try:
        out["auroc_norm_vs_label"] = float(eval_roc_auc(y_bool, norms.cpu()))
    except Exception as e:
        out["auroc_norm_vs_label"] = None

    # interpretation helpers
    out["interpret_r_eff"] = (
        "TRUE_COLLAPSE_NEAR_1D" if out["r_eff"] < 1.5 else
        "LOW_RANK" if out["r_eff"] < 3.0 else
        "MULTI_DIM_OK"
    )
    if out["std_cos_to_center"] < 1e-4 and out["std_norm"] > 1e-3:
        out["interpret_dir_norm"] = "DIRECTIONAL_COLLAPSE_NORM_VARIES"
    elif out["std_cos_to_center"] < 1e-4 and out["std_norm"] < 1e-3:
        out["interpret_dir_norm"] = "FULL_COLLAPSE_DIR_AND_NORM"
    else:
        out["interpret_dir_norm"] = "COSINE_HAS_SPREAD"
    if out["std_euclid_center"] > 1e-4 and out["std_cos_to_center"] < 1e-4:
        out["interpret_metric"] = "SIMILARITY_CHOICE_PROBLEM"
    elif out["std_euclid_center"] < 1e-4 and out["std_cos_to_center"] < 1e-4:
        out["interpret_metric"] = "GAE_EMBEDDING_PROBLEM"
    else:
        out["interpret_metric"] = "BOTH_OR_COSINE_OK"

    return out


def _get_frozen_H(dataset: str, seed: int, gpu: int, out_root: Path) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    os.environ["FMGAD_RUN_TAG_SUFFIX"] = f"embed_diag_s{seed}"
    _set_seed(seed, deterministic=True)

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    model = ResFlowGAD(
        hid_dim=None,
        ae_dropout=float(cfg["ae_dropout"]),
        ae_lr=float(cfg["ae_lr"]),
        ae_alpha=float(cfg["ae_alpha"]),
        use_proto=False,
        dual_flow_mode=None,
        residual_scale=float(cfg["residual_scale"]),
        sample_steps=1,
        verbose=False,
        num_trial=1,
        exp_tag=f"embed_diag_{dataset}",
        score_smoothing_alpha=float(cfg.get("score_smoothing_alpha", 0.3)),
        polarity_enabled=False,
        use_virtual_neighbors=bool(cfg.get("use_virtual_neighbors", True)),
        virtual_degree_threshold=int(cfg.get("virtual_degree_threshold", 5)),
        virtual_k=int(cfg.get("virtual_k", 5)),
    )
    data = load_data(dataset)
    if model.hid_dim is None:
        import math

        model.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)
    from fmgad.models.autoencoder import GraphAE

    model.ae = GraphAE(
        in_dim=data.num_node_features, hid_dim=model.hid_dim, dropout=model.ae_dropout
    ).cuda()
    save_dir = model._ensure_save_dir(dataset)
    ae_path = os.path.join(
        save_dir,
        f"ae_drop{model.ae_dropout}_lr{model.ae_lr}_alpha{model.ae_alpha}_hid{model.hid_dim}",
        f"embed_diag_{dataset}_s{seed}",
    )
    os.makedirs(ae_path, exist_ok=True)
    ae_ckpt = model._train_ae_once(data, ae_path)
    ae_dict = torch.load(os.path.join(ae_path, f"{ae_ckpt}.pt"), map_location="cuda")
    model.ae.load_state_dict(ae_dict["state_dict"])
    model._freeze_ae_eval()

    x = data.x.cuda().float()
    edge_index = data.edge_index.cuda()
    with torch.no_grad():
        # Match center_dom cache path: encode via _build_z so virtual edges apply.
        z, h, r = model._build_z(x, edge_index)
    return {
        "H": h.detach().cpu(),
        "R": r.detach().cpu(),
        "Z": z.detach().cpu(),
        "y": data.y.detach().cpu(),
        "ae_ckpt": int(ae_ckpt),
        "hid_dim": int(model.hid_dim),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--output-dir", type=str, default="results/embed_collapse_diag")
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    gpu = gpus[0]

    all_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        for seed in seeds:
            print(f"=== {ds} seed={seed} ===", flush=True)
            pack = _get_frozen_H(ds, seed, gpu, out_root)
            diag_h = _diagnose_H(pack["H"], pack["y"])
            diag_r = _diagnose_H(pack["R"], pack["y"])
            row = {
                "dataset": ds,
                "seed": seed,
                "ae_ckpt": pack["ae_ckpt"],
                "hid_dim": pack["hid_dim"],
                "H": diag_h,
                "R_residual": diag_r,
            }
            all_rows.append(row)
            print(
                f"  H: r_eff={diag_h['r_eff']:.3f} std_cos={diag_h['std_cos_to_center']:.3g} "
                f"std_norm={diag_h['std_norm']:.3g} std_euclid={diag_h['std_euclid_center']:.3g} "
                f"| {diag_h['interpret_r_eff']} / {diag_h['interpret_dir_norm']} / {diag_h['interpret_metric']}",
                flush=True,
            )
            print(
                f"     AUROC cos={diag_h['auroc_neg_cos_vs_label']} "
                f"euclid={diag_h['auroc_euclid_vs_label']} norm={diag_h['auroc_norm_vs_label']}",
                flush=True,
            )
            print(
                f"  R: r_eff={diag_r['r_eff']:.3f} std_cos={diag_r['std_cos_to_center']:.3g} "
                f"std_euclid={diag_r['std_euclid_center']:.3g} "
                f"AUROC euclid={diag_r['auroc_euclid_vs_label']}",
                flush=True,
            )

    # markdown summary
    lines = [
        "# Embedding collapse diagnostics (analysis only)",
        "",
        "Protocol: AE trained then `eval()`+freeze; no flow training.",
        "",
        "## Summary table (H = AE embedding)",
        "",
        "| Dataset | Seed | r_eff | std(cos) | std(||h||) | mean pairwise cos | std(d_euclid) | AUROC(-cos)† | AUROC(d)† | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in all_rows:
        h = row["H"]
        verdict = h["interpret_metric"]
        if h["interpret_r_eff"] == "TRUE_COLLAPSE_NEAR_1D":
            verdict = "TRUE_COLLAPSE / " + verdict
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {h['r_eff']:.3f} | "
            f"{h['std_cos_to_center']:.3g} | {h['std_norm']:.3g} | "
            f"{h['mean_pairwise_cos']:.4f} | {h['std_euclid_center']:.3g} | "
            f"{h['auroc_neg_cos_vs_label']} | {h['auroc_euclid_vs_label']} | "
            f"{h['interpret_dir_norm']}; {verdict} |"
        )

    lines += [
        "",
        "† Label AUROCs are **post-hoc analysis only**, not for HP selection.",
        "",
        "## Residual R (for reference)",
        "",
        "| Dataset | Seed | r_eff | std(cos) | std(d_euclid) | AUROC(d)† |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        r = row["R_residual"]
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {r['r_eff']:.3f} | "
            f"{r['std_cos_to_center']:.3g} | {r['std_euclid_center']:.3g} | "
            f"{r['auroc_euclid_vs_label']} |"
        )

    lines += [
        "",
        "## How to read",
        "",
        "- `r_eff ≈ 1` → representation nearly 1-D (true collapse).",
        "- cos flat but `std(||h||)` large → directional collapse; norms still vary.",
        "- Euclidean `d_i` discriminative but cos not → **similarity choice** problem.",
        "- Neither discriminative → **GAE embedding** problem.",
        "",
    ]

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    # drop tensors for json
    json.dump(all_rows, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)
    print(f"\nWrote {summary_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
