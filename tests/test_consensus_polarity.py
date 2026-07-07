import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import calibrate_polarity_consensus_rank


def test_consensus_keeps_strongly_aligned_score():
    score = torch.tensor([0.1, 0.3, 0.2, 0.9, 0.7])
    probes = [score * 10.0 + 2.0, score.square()]
    out, flipped, diag = calibrate_polarity_consensus_rank(score, probes)
    assert not flipped
    assert diag["decision"] == "keep"
    assert diag["agreement"] >= 0.70
    assert torch.argsort(out).tolist() == torch.argsort(score).tolist()


def test_consensus_flips_disagreeing_score():
    score = torch.tensor([0.1, 0.3, 0.2, 0.9, 0.7])
    inverse = 1.0 - score
    out, flipped, diag = calibrate_polarity_consensus_rank(score, [inverse, inverse * 3.0])
    assert flipped
    assert diag["decision"] == "flip"
    assert torch.argsort(out).tolist() == torch.argsort(inverse).tolist()


def test_consensus_is_invariant_to_positive_probe_scaling():
    score = torch.tensor([4.0, 1.0, 3.0, 2.0])
    probe = torch.tensor([1.0, 4.0, 2.0, 3.0])
    out_a, flip_a, _ = calibrate_polarity_consensus_rank(score, [probe])
    out_b, flip_b, _ = calibrate_polarity_consensus_rank(score, [probe * 1000.0 + 17.0])
    assert flip_a == flip_b
    assert torch.equal(out_a, out_b)


if __name__ == "__main__":
    test_consensus_keeps_strongly_aligned_score()
    test_consensus_flips_disagreeing_score()
    test_consensus_is_invariant_to_positive_probe_scaling()
    print("consensus polarity tests passed")
