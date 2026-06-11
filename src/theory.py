"""Closed-form theory and the empirical packing-law fit protocol.

Sources:
- Elhage et al. 2022, "Toy Models of Superposition" (model, Zipf importances).
- Scherlis et al. 2022, "Polysemanticity and Capacity in Neural Networks"
  (fractional capacity C_i; see `src.metrics.feature_capacity`).
- Sarkar & Deka 2026, "Geometric Limits of Knowledge Distillation"
  (arXiv:2604.04037): capacity function g(alpha), critical width d*, and
  the predicted loss floor of their Eq. 2.

Data distribution assumed throughout: x_i ~ U[0,1] with probability
(1 - alpha), else 0, iid across features (alpha = sparsity).

Notation follows the paper: `g_alpha` for g(alpha), `d_star` for the
critical width, `F_kept` for the kept-feature count floor(d * g(alpha)).
"""

import numpy as np


def g_alpha(alpha: float) -> float:
    """Capacity function g(alpha) = 1 / ((1-alpha) * ln(1/(1-alpha))).

    Number of features a single hidden dimension can encode at sparsity
    alpha; a width-d model encodes <= d * g(alpha) features (their Thm. 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    p = 1.0 - alpha  # activation probability
    return 1.0 / (p * np.log(1.0 / p))


def d_star(n: int, alpha: float) -> float:
    """Critical width d* = n / g(alpha): width above which all n features fit."""
    return n / g_alpha(alpha)


def F_kept(d: int, alpha: float, n: int) -> int:
    """Eq. 2 kept-feature count F = floor(d * g(alpha)), capped at n."""
    return min(int(np.floor(d * g_alpha(alpha))), n)


def power_law_importances(n: int, s: float) -> np.ndarray:
    """I_i proportional to i^(-s) for i = 1..n; s=0 is uniform, s=1 is Zipf."""
    return 1.0 / np.arange(1, n + 1) ** s


def zipf_importances(n: int) -> np.ndarray:
    """I_i = 1/i for i = 1..n (unnormalized, sorted descending)."""
    return power_law_importances(n, 1.0)


def expected_x_sq(alpha: float) -> float:
    """E[x_i^2] = (1-alpha)/3 for the sparse-uniform distribution.

    This is the per-feature floor used by Eq. 2 of the paper (it charges
    a dropped feature E[x^2], i.e. assumes the model outputs 0 for it; a
    bias-optimal constant output would achieve Var(x_i), slightly less,
    so trained models can land slightly below the per-feature prediction).
    """
    return (1.0 - alpha) / 3.0


def predicted_floor(
    d: int,
    n: int,
    alpha: float,
    importances: np.ndarray | None = None,
    order_by: np.ndarray | None = None,
    F: int | None = None,
) -> float:
    """Eq. 2 loss floor: L*(d) = sum over dropped features of I_i * E[x^2].

    The F features with the largest `order_by` value are kept; every
    dropped feature costs its TRUE importance times E[x^2].

    importances: true importances I (defaults to Zipf).
    order_by: ranking used for the keep decision (defaults to I; assumption
        A2). Exp B passes boosted weights I-tilde to predict the floor of
        controlled placement.
    F: kept count (defaults to F_kept(d, alpha, n), the paper's Eq. 2;
        the equilibrium-corrected predictor passes round(d * ghat(alpha))).
    """
    I = zipf_importances(n) if importances is None else np.asarray(importances, dtype=float)
    keys = I if order_by is None else np.asarray(order_by, dtype=float)
    F = F_kept(d, alpha, n) if F is None else F
    if F >= n:
        return 0.0
    dropped = np.argsort(-keys, kind="stable")[max(F, 0):]
    return float(I[dropped].sum() * expected_x_sq(alpha))


def fit_packing_law(g_values: np.ndarray, ghat_values: np.ndarray) -> dict:
    """Fit the empirical packing law ghat(alpha) = a * g(alpha)^b.

    THE canonical fit protocol used for every packing-law number in this
    repository (README "science harmonization" note):

      1. Train the slope-law probe (`experiments/probe_slope_law.py`) at
         n=200, d=10, 3 seeds, update-equalized to 1e6 active samples per
         feature, over alpha in {0.80, 0.90, 0.95, 0.99}.
      2. Count kept features as ||W_i||^2 > 0.5 (threshold-robust across
         tau in [0.3, 0.7]; see the report's robustness table).
      3. Average counts over seeds per alpha; ghat = mean count / d.
      4. Least-squares fit of log(ghat) on log(g) over the alpha grid.

    An earlier, independent 3-seed run of the same protocol (the capacity
    probe) gave b = 0.57 for Zipf vs 0.53 here — within seed noise; the
    slope-law instrument is canonical because it stores per-feature norms,
    enabling the threshold-robustness check.

    Returns {"a": float, "b": float}; NaN if fewer than 2 valid points.
    """
    g = np.asarray(g_values, dtype=float)
    ghat = np.asarray(ghat_values, dtype=float)
    ok = (g > 0) & (ghat > 0) & np.isfinite(ghat)
    if ok.sum() < 2:
        return {"a": float("nan"), "b": float("nan")}
    b, log_a = np.polyfit(np.log(g[ok]), np.log(ghat[ok]), 1)
    return {"a": float(np.exp(log_a)), "b": float(b)}
