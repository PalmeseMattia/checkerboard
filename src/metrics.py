"""Representation metrics: feature survival and allocation order."""

import numpy as np
import torch

SURVIVAL_THRESHOLD = 0.5  # ||W[:, i]||^2 above this => feature i is represented


def column_norms_sq(W: torch.Tensor) -> torch.Tensor:
    """||W[:, i]||^2 per feature. W: (..., d, n) -> (..., n)."""
    return (W**2).sum(dim=-2)


def survived_mask(col_norms_sq: torch.Tensor, threshold: float = SURVIVAL_THRESHOLD) -> torch.Tensor:
    """Survived set S as a boolean mask over features."""
    return col_norms_sq > threshold


def feature_capacity(W: torch.Tensor) -> torch.Tensor:
    """C_i = ||W_i||^4 / sum_j (W_i . W_j)^2 per feature column. (..., d, n) -> (..., n).

    Scherlis et al. capacity, identical to Elhage et al. feature
    dimensionality: 1 for a feature alone in an orthogonal direction,
    1/2 for an antipodal pair, 0 for a dead feature. Satisfies
    sum_i C_i <= d, so sum_i C_i / d measures decoder capacity
    utilization independent of any survival threshold.
    """
    G = torch.einsum("...di,...dj->...ij", W, W)  # column Gram matrix
    num = torch.diagonal(G, dim1=-2, dim2=-1) ** 2
    den = (G**2).sum(dim=-1)
    return torch.where(den > 1e-12, num / den, torch.zeros_like(num))


def overlap_at_k(survived: np.ndarray, importances: np.ndarray) -> float:
    """|S intersect top-|S|-by-importance| / |S|; NaN if S is empty.

    Equals 1.0 iff the model kept exactly the |S| most important
    features (assumption A2); < 1.0 quantifies the violation.
    """
    survived = np.asarray(survived, dtype=bool)
    k = int(survived.sum())
    if k == 0:
        return float("nan")
    top_k = np.argsort(-np.asarray(importances, dtype=float), kind="stable")[:k]
    return float(np.intersect1d(np.flatnonzero(survived), top_k).size) / k
