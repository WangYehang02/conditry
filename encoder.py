import torch


def compute_residuals(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    AnomalyGFM-style residual:
      r_i = h_i - mean_{j in N(i)} h_j

    edge_index is PyG format [2, E]; messages aggregate from src=edge_index[0] to dst=edge_index[1].
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)

    # sum_{j->i} h_j
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    deg = deg.clamp_min(1.0).unsqueeze(1)  # avoid div-by-zero; isolated nodes treated as zero neighbor mean

    neigh_mean = neigh_sum / deg
    return h - neigh_mean


def compute_dual_residuals_with_degree(h: torch.Tensor, edge_index: torch.Tensor):
    """
    Dual residuals (global + local) and node degree for adaptive gating.
    Global residual: deviation from the graph-wide mean (robust on sparse / low-degree nodes).
    Local residual: deviation from neighbor mean (sensitive to local structure on dense graphs).
    edge_index is PyG [2, E]; aggregate src -> dst.

    Returns:
        r_global: [N, D] global statistical residual
        r_local:  [N, D] local structural residual
        deg:      [N, 1] node degree (for gating)
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    # 1) Global residual: deviation from graph-wide mean
    global_mean = torch.mean(h, dim=0, keepdim=True)
    r_global = h - global_mean

    # 2) Local residual and degree
    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg_val = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))

    deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
    neigh_mean = neigh_sum / deg_clamped
    r_local = h - neigh_mean

    return r_global, r_local, deg_val.unsqueeze(1)
