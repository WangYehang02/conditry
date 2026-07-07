"""Scoring helpers for FMGAD local-prior polarity calibration."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unit_rank_tensor(value: torch.Tensor) -> torch.Tensor:
    """Deterministic ordinal ranks in [0, 1], preserving only score order."""
    flat = value.detach().reshape(-1).float()
    n = int(flat.numel())
    if n <= 1:
        return torch.zeros_like(flat)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty(n, device=flat.device, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=flat.device, dtype=torch.float32)
    return ranks / float(n - 1)


def calibrate_polarity_consensus_rank(
    score: torch.Tensor,
    probes: List[torch.Tensor],
    *,
    agreement_threshold: float = 0.70,
    score_weight: float = 0.90,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """Orient the main score using rank agreement with local_prior (or other directed probes)."""
    score_rank = _unit_rank_tensor(score)
    valid = [p for p in probes if p is not None and int(p.numel()) == int(score_rank.numel())]
    if not valid:
        return score, False, {
            "mode": "local_prior_rank",
            "decision": "abstain",
            "flipped": False,
            "reason": "no_valid_probes",
        }

    probe_ranks = torch.stack([_unit_rank_tensor(p.to(score_rank.device)) for p in valid], dim=0)
    consensus = probe_ranks.mean(dim=0)
    a = score_rank - score_rank.mean()
    b = consensus - consensus.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    agreement = float(torch.dot(a, b) / denom) if float(denom) > 1e-12 else 0.0
    flipped = agreement < float(agreement_threshold)
    oriented = 1.0 - score_rank if flipped else score_rank
    weight = float(np.clip(score_weight, 0.0, 1.0))
    calibrated = weight * oriented + (1.0 - weight) * consensus
    return calibrated.to(score.device), flipped, {
        "mode": "local_prior_rank",
        "decision": "flip" if flipped else "keep",
        "flipped": bool(flipped),
        "agreement": agreement,
        "agreement_threshold": float(agreement_threshold),
        "score_weight": weight,
        "num_probes": len(valid),
    }


def softmax_with_temperature(x: torch.Tensor, t: float = 1.0, dim: int = -1) -> torch.Tensor:
    return F.softmax(x / t, dim=dim)


def compute_local_prior(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-node ||x_i - mean(neighbors(i))||_2 — the sole polarity probe."""
    with torch.no_grad():
        xf = x.float()
        src, dst = edge_index[0], edge_index[1]
        n = xf.size(0)
        neigh_sum = torch.zeros_like(xf)
        neigh_sum.index_add_(0, dst, xf[src])
        deg = torch.zeros(n, device=xf.device, dtype=xf.dtype)
        deg.index_add_(0, dst, torch.ones_like(xf[src, 0]))
        deg_u = deg.clamp_min(1.0).unsqueeze(-1)
        neigh_mean = neigh_sum / deg_u
        return torch.norm(xf - neigh_mean, p=2, dim=1)


# Backward-compatible name used by the original scripts.
compute_smoothgnn_local_prior = compute_local_prior
