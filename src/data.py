"""Sparse feature data generation (Elhage et al. 2022 setup)."""

import torch


def sample_features_blocked(
    shape: tuple[int, ...],
    n: int,
    alpha: float,
    group_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Block-anticorrelated features: at most one active per group.

    Features are partitioned into consecutive groups of `group_size`;
    each group activates with probability group_size * (1 - alpha) and
    a uniformly chosen member takes a U[0,1] value. Per-feature
    marginals match the iid sampler exactly — P(active) = 1 - alpha and
    E[x^2] = (1 - alpha)/3 — so Eq. 2's per-feature floor is unchanged;
    only the correlation structure differs (within-group co-activation
    probability is exactly 0 instead of (1 - alpha)^2).
    """
    if n % group_size:
        raise ValueError(f"n={n} not divisible by group_size={group_size}")
    p_group = group_size * (1.0 - alpha)
    if p_group > 1.0:
        raise ValueError(f"group_size*(1-alpha)={p_group:.2f} > 1: not satisfiable")
    G = n // group_size
    size = (*shape, G)
    active = torch.rand(size, generator=generator, device=device) < p_group
    u = torch.rand(size, generator=generator, device=device)
    member = (
        torch.rand(size, generator=generator, device=device) * group_size
    ).long().clamp_(max=group_size - 1)
    onehot = member.unsqueeze(-1) == torch.arange(group_size, device=device)
    x = (u * active).unsqueeze(-1) * onehot
    return x.reshape(*shape, n)


def sample_features(
    shape: tuple[int, ...],
    n: int,
    alpha: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample x of shape (*shape, n): x_i ~ U[0,1] w.p. (1-alpha), else 0, iid."""
    size = (*shape, n)
    u = torch.rand(size, generator=generator, device=device)
    active = torch.rand(size, generator=generator, device=device) >= alpha
    return u * active
