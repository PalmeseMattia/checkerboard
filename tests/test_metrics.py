import math

import numpy as np
import torch

from src.metrics import column_norms_sq, feature_capacity, overlap_at_k, survived_mask


def test_column_norms_sq():
    W = torch.tensor([[[1.0, 0.0], [2.0, 3.0]]])  # (1, d=2, n=2)
    assert torch.allclose(column_norms_sq(W), torch.tensor([[5.0, 9.0]]))


def test_survived_mask_threshold():
    cns = torch.tensor([0.9, 0.4, 0.51])
    assert survived_mask(cns).tolist() == [True, False, True]


def test_feature_capacity_orthogonal():
    # Two features in orthogonal directions: C_i = 1 each, sum = d.
    W = torch.eye(2).unsqueeze(0)  # (1, d=2, n=2)
    assert torch.allclose(feature_capacity(W), torch.ones(1, 2))


def test_feature_capacity_antipodal_pair():
    # Antipodal superposition in one dimension: C_i = 1/2, sum = d = 1.
    W = torch.tensor([[[1.0, -1.0]]])  # (1, d=1, n=2)
    assert torch.allclose(feature_capacity(W), torch.full((1, 2), 0.5))


def test_feature_capacity_dead_feature_and_bound():
    W = torch.tensor([[[1.0, 0.7, 0.0], [0.0, 0.7, 0.0]]])  # (1, d=2, n=3)
    C = feature_capacity(W)
    assert C[0, 2] == 0.0  # dead feature
    assert C.sum() <= 2.0 + 1e-6  # sum_i C_i <= d


def test_overlap_at_k():
    I = np.array([1.0, 0.5, 1 / 3, 0.25])
    # Kept exactly the top 2: perfect overlap.
    assert overlap_at_k(np.array([True, True, False, False]), I) == 1.0
    # Kept {0, 3} instead of {0, 1}: one of two misplaced.
    assert overlap_at_k(np.array([True, False, False, True]), I) == 0.5
    # Empty survived set is undefined.
    assert math.isnan(overlap_at_k(np.array([False] * 4), I))
