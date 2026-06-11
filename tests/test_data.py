import pytest
import torch

from src.data import sample_features, sample_features_blocked


def test_blocked_marginals_and_exclusivity():
    n, alpha, k = 16, 0.9, 4
    gen = torch.Generator().manual_seed(0)
    x = sample_features_blocked((50_000,), n, alpha, k, gen)
    assert x.shape == (50_000, n)
    # At most one active feature per group, always.
    per_group = (x.reshape(-1, n // k, k) > 0).sum(-1)
    assert per_group.max() <= 1
    # Per-feature marginals match the iid sampler: P(active) = 1 - alpha.
    p_active = (x > 0).float().mean(0)
    assert torch.allclose(p_active, torch.full((n,), 1 - alpha), atol=0.01)
    # E[x^2] = (1 - alpha) / 3.
    assert torch.allclose(x.pow(2).mean(0), torch.full((n,), (1 - alpha) / 3), atol=0.005)


def test_blocked_rejects_unsatisfiable_config():
    with pytest.raises(ValueError):
        sample_features_blocked((8,), 16, 0.5, 4)  # group prob 2.0 > 1
    with pytest.raises(ValueError):
        sample_features_blocked((8,), 15, 0.9, 4)  # n not divisible


def test_shape_and_range():
    x = sample_features((3, 128), n=40, alpha=0.9, generator=torch.Generator().manual_seed(0))
    assert x.shape == (3, 128, 40)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_sparsity_level():
    alpha = 0.9
    x = sample_features((200_000,), n=10, alpha=alpha, generator=torch.Generator().manual_seed(1))
    zero_frac = (x == 0).float().mean().item()
    assert abs(zero_frac - alpha) < 0.005


def test_deterministic_given_seed():
    a = sample_features((64,), n=8, alpha=0.5, generator=torch.Generator().manual_seed(42))
    b = sample_features((64,), n=8, alpha=0.5, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
