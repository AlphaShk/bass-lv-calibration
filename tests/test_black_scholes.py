"""Remark 2.6: with lognormal marginals of a common volatility, Bass-LV is Black-Scholes."""

import numpy as np
from scipy.stats import norm

import bass_lv as bl

from helpers import FLAT_MATURITIES, SIGMA

W = bl.W_GRID


def gbm_map(t, w):
    return np.exp(-0.5 * SIGMA**2 * t + SIGMA * w)


def test_classical_bass_interval_is_geometric_brownian_motion(flat_marginals):
    T1 = FLAT_MATURITIES[0]
    first = bl.classical_interval(flat_marginals[0], T1)
    central = np.abs(W) <= 1.0  # two standard deviations of W_{T_1}
    for t in (0.0, T1 / 2, T1):
        assert np.max(np.abs(first.mapping(t)[central] / gbm_map(t, W[central]) - 1)) < 1e-3
    assert abs(first.mapping(0.0, np.array(0.0)) - 1) < 1e-4  # S_0 = 1
    S, sigma = first.local_vol(T1 / 2)
    keep = (S > 0.85) & (S < 1.15)  # beyond ~2 s.d. the quantile grid's tail cut-off shows
    assert np.max(np.abs(sigma[keep] - SIGMA)) < 1e-3


def test_lognormal_fixed_point_is_the_brownian_law(flat_marginals):
    T1, T2 = FLAT_MATURITIES[1], FLAT_MATURITIES[2]
    pair = bl.calibrate_pair(flat_marginals[1], flat_marginals[2], T1, T2)
    assert pair.errors[-1] < 1e-5
    assert np.max(np.abs(pair.latent_cdf - norm.cdf(W / np.sqrt(T1)))) < 5e-4
    central = np.abs(W) <= 3 * np.sqrt(T1)
    assert np.max(np.abs(pair.terminal[central] / gbm_map(T2, W[central]) - 1)) < 5e-4
    for t in (T1, (T1 + T2) / 2, T2 - 0.05):
        S, sigma = pair.local_vol(t)
        keep = (S > 0.8) & (S < 1.25)
        assert np.max(np.abs(sigma[keep] - SIGMA)) < 1e-3


def test_gaussian_initializer_is_already_a_fixed_point(flat_marginals):
    T1, T2 = FLAT_MATURITIES[1], FLAT_MATURITIES[2]
    pair = bl.calibrate_pair(flat_marginals[1], flat_marginals[2], T1, T2, initial="gaussian")
    assert pair.iterations <= 5


def test_simulated_returns_are_black_scholes(flat_model):
    n = 400_000
    X = bl.simulate_market_dates(flat_model, n=n, seed=2)
    for i, dt in enumerate(np.diff(FLAT_MATURITIES)):
        r = np.log(X[:, i + 1] / X[:, i])
        sd = SIGMA * np.sqrt(dt)
        assert abs(r.mean() + 0.5 * sd**2) < 4 * sd / np.sqrt(n)
        assert abs(r.std() / sd - 1) < 4 / np.sqrt(2 * n)
        assert abs(np.corrcoef(np.log(X[:, i]), r)[0, 1]) < 4 / np.sqrt(n)
        ratio = X[:, i + 1] / X[:, i]
        for k in (0.9, 1.0, 1.1):  # forward-start calls, Section 4.2.1
            payoff = np.maximum(ratio - k, 0)
            se = payoff.std(ddof=1) / np.sqrt(n)
            assert abs(payoff.mean() - bl.black_call(k, SIGMA, dt)) < 4 * se


def test_asian_price_matches_black_scholes_monte_carlo(flat_model):
    n = 200_000
    X = bl.simulate_market_dates(flat_model, n=n, seed=3)
    bass, bass_se = bl.asian_call(X, np.ones(3), 1.0, 1.0)
    rng = np.random.default_rng(4)
    increments = rng.standard_normal((n, 3)) * SIGMA * np.sqrt(np.diff(np.r_[0, FLAT_MATURITIES]))
    log_paths = np.cumsum(increments, axis=1) - 0.5 * SIGMA**2 * FLAT_MATURITIES
    bs, bs_se = bl.asian_call(np.exp(log_paths), np.ones(3), 1.0, 1.0)
    assert abs(bass - bs) < 4 * np.hypot(bass_se, bs_se)
