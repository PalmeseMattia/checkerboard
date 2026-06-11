import numpy as np
import pytest

from src.theory import (
    F_kept,
    d_star,
    expected_x_sq,
    fit_packing_law,
    g_alpha,
    power_law_importances,
    predicted_floor,
    zipf_importances,
)


def test_g_alpha_closed_form():
    # g(0.90) = 1 / (0.1 * ln 10)
    assert g_alpha(0.90) == pytest.approx(1.0 / (0.1 * np.log(10.0)))
    # g(0.99) = 100 / ln 100
    assert g_alpha(0.99) == pytest.approx(100.0 / np.log(100.0))


def test_g_alpha_rejects_degenerate_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            g_alpha(bad)


def test_d_star():
    # d* = n / g(alpha) = n * (1-alpha) * ln(1/(1-alpha))
    assert d_star(40, 0.90) == pytest.approx(40 * 0.1 * np.log(10.0))


def test_F_kept_caps_at_n():
    assert F_kept(100, 0.90, 40) == 40  # floor(100*g) > n, capped
    assert F_kept(9, 0.90, 40) == 39    # floor(9 * 4.3429)


def test_power_law_and_zipf_agree():
    assert np.allclose(power_law_importances(10, 1.0), zipf_importances(10))
    assert np.allclose(power_law_importances(5, 0.0), np.ones(5))  # uniform, s=0


def test_fit_packing_law_recovers_known_exponent():
    g = np.array([2.0, 4.0, 8.0, 16.0])
    ghat = 0.9 * g ** 0.6
    fit = fit_packing_law(g, ghat)
    assert fit["a"] == pytest.approx(0.9, abs=1e-6)
    assert fit["b"] == pytest.approx(0.6, abs=1e-6)


def test_fit_packing_law_degenerate_returns_nan():
    fit = fit_packing_law(np.array([2.0]), np.array([1.0]))
    assert np.isnan(fit["a"]) and np.isnan(fit["b"])


def test_zipf_importances():
    I = zipf_importances(4)
    assert np.allclose(I, [1.0, 0.5, 1.0 / 3.0, 0.25])
    assert np.all(np.diff(I) < 0)


def test_expected_x_sq():
    # E[x^2] = (1-alpha) * int_0^1 u^2 du = (1-alpha)/3
    assert expected_x_sq(0.90) == pytest.approx(0.1 / 3.0)


def test_predicted_floor_monotone_nonincreasing():
    n, alpha = 40, 0.90
    floors = [predicted_floor(d, n, alpha) for d in range(0, 12)]
    assert all(a >= b for a, b in zip(floors, floors[1:]))


def test_predicted_floor_endpoints():
    n, alpha = 40, 0.90
    I = zipf_importances(n)
    # d = 0: everything dropped.
    assert predicted_floor(0, n, alpha) == pytest.approx(I.sum() * expected_x_sq(alpha))
    # Above the critical width all features fit: floor is exactly 0.
    assert predicted_floor(int(np.ceil(d_star(n, alpha))) + 1, n, alpha) == 0.0


def test_predicted_floor_hand_computed():
    # n=40, alpha=0.90: F = floor(9 * 4.3429...) = 39, only feature 40 dropped.
    n, alpha = 40, 0.90
    expected = (1.0 / 40.0) * expected_x_sq(alpha)
    assert predicted_floor(9, n, alpha) == pytest.approx(expected)


def test_predicted_floor_with_placement_ordering():
    # n=4, keep F=2. Boosting the least-important feature keeps {1, 4}
    # and drops {2, 3}; the cost is charged at TRUE importances.
    n, alpha, d = 4, 0.50, 1  # g(0.5) = 1/(0.5 ln 2) ~ 2.885 -> F = 2
    assert F_kept(d, alpha, n) == 2
    I = zipf_importances(n)
    boosted = I.copy()
    boosted[3] *= 100.0
    vanilla = predicted_floor(d, n, alpha, importances=I)
    placed = predicted_floor(d, n, alpha, importances=I, order_by=boosted)
    assert vanilla == pytest.approx((I[2] + I[3]) * expected_x_sq(alpha))
    assert placed == pytest.approx((I[1] + I[2]) * expected_x_sq(alpha))
    assert placed > vanilla  # placement has a predictable cost


def test_predicted_floor_custom_count():
    # Equilibrium predictor: a smaller kept count F drops more importance.
    n, alpha, d = 40, 0.90, 5
    big = predicted_floor(d, n, alpha, F=F_kept(d, alpha, n))
    small = predicted_floor(d, n, alpha, F=F_kept(d, alpha, n) - 5)
    assert small > big
