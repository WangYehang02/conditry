import torch

from fmgad.graph_ops import add_virtual_knn_edges


def test_virtual_knn_adds_incoming_edges_for_low_degree_nodes():
    # Node 0 is isolated; nodes 1-3 form a triangle. h makes 1,2 closest to 0.
    edge_index = torch.tensor(
        [[1, 2, 3], [2, 3, 1]],
        dtype=torch.long,
    )
    h = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    out = add_virtual_knn_edges(edge_index, h, degree_threshold=2, k=2, device=h.device)

    # Node 0 is isolated; supplementation must add incoming edges (j -> 0).
    to_zero = out[:, out[1] == 0]
    assert to_zero.numel() > 0
    assert (to_zero[0] != 0).all()
    assert int(to_zero.size(1)) == 2


def test_virtual_knn_supplements_up_to_threshold():
    # Two isolated nodes; threshold=2, k=2 => add 2 incoming edges each.
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    h = torch.eye(4)
    out = add_virtual_knn_edges(edge_index, h, degree_threshold=2, k=2, device=h.device)
    in_deg = torch.zeros(4, dtype=torch.long)
    in_deg.scatter_add_(0, out[1], torch.ones(out.size(1), dtype=torch.long))
    assert int(in_deg[0]) == 2
    assert int(in_deg[1]) == 2


if __name__ == "__main__":
    test_virtual_knn_adds_incoming_edges_for_low_degree_nodes()
    test_virtual_knn_supplements_up_to_threshold()
    print("virtual neighbor tests passed")
