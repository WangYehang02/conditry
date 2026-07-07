"""Small graph operations shared by training and scoring."""

import torch


def smooth_scores_by_graph(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Blend a score with the mean score of incoming neighbors."""
    if alpha <= 0.0 or edge_index.numel() == 0:
        return score
    src, dst = edge_index[0], edge_index[1]
    n = score.size(0)
    neigh_sum = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_sum.index_add_(0, dst, score[src])
    deg = torch.zeros(n, device=device, dtype=score.dtype)
    deg.index_add_(0, dst, torch.ones_like(score[src]))
    neigh_mean = neigh_sum / deg.clamp_min(1.0)
    return (1.0 - alpha) * score + alpha * neigh_mean


def add_virtual_knn_edges(
    edge_index: torch.Tensor,
    h: torch.Tensor,
    degree_threshold: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Append embedding-space kNN edges for low-degree nodes."""
    n = h.size(0)
    if n > 50000:
        return edge_index
    with torch.no_grad():
        deg = torch.zeros(n, device=device, dtype=torch.long)
        deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device, dtype=torch.long))
        low_deg_mask = deg < degree_threshold
        if low_deg_mask.sum() == 0:
            return edge_index

        h_norm = torch.nn.functional.normalize(h, p=2, dim=1)
        sim = torch.mm(h_norm, h_norm.t())
        sim.fill_diagonal_(-1e9)
        _, idx = sim.topk(min(k, n - 1), dim=1)

        new_edges = []
        for i in range(n):
            if not low_deg_mask[i]:
                continue
            for j in idx[i].tolist():
                if j != i:
                    new_edges.append([i, j])
        if not new_edges:
            return edge_index

        new_edges = torch.tensor(new_edges, device=device, dtype=edge_index.dtype).t()
        combined = torch.cat([edge_index, new_edges], dim=1)
        return torch.unique(combined, dim=1)
