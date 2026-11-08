"""Mean-one marginals and the martingale property of the calibrated model."""

import numpy as np
import pytest

import bass_lv as bl

from helpers import SKEW_MATURITIES, skew_total_variance


def conditional_drift_z(paths, bins=10):
    """z-scores of E[X_{i+1} - X_i | X_i in decile] for every interval and decile."""
    table = bl.martingale_diagnostic(paths, bins)
    assert len(table) == bins * (paths.shape[1] - 1)
    return table["z"].to_numpy()


def test_marginals_have_mean_one(skew_marginals, flat_marginals):
    for marginal in skew_marginals + flat_marginals:
        assert abs(marginal.mean() - 1) < 1e-12


@pytest.mark.parametrize("T", SKEW_MATURITIES)
def test_breeden_litzenberger_marginal_reprices_the_smile(T):
    marginal = bl.marginal_from_total_variance(skew_total_variance(bl.X_GRID, T), bl.X_GRID)
    for k in (0.85, 0.9, 1.0, 1.1, 1.15):
        vol = np.sqrt(skew_total_variance(np.log(k), T) / T)
        assert abs(marginal.call(k) - bl.black_call(k, vol, T)) < 5e-4


def test_maps_push_the_latent_law_onto_the_marginals(skew_model, skew_marginals):
    """Appendix identities: F_mu1(f(T1, w)) = F_{W_T1}(w), F_mu2(f(T2-, w)) = F_{W_T2-}(w)."""
    for i, pair in enumerate(skew_model.pairs):
        dt = pair.t1 - pair.t0
        assert np.max(np.abs(skew_marginals[i].cdf(pair.start) - pair.latent_cdf)) < 3e-5
        pre = bl.heat(pair.latent_cdf, bl.W_GRID, dt, cdf=True)
        assert np.max(np.abs(skew_marginals[i + 1].cdf(pair.terminal) - pre)) < 2e-5


def test_start_map_is_the_heat_flow_of_the_terminal_map(skew_model):
    for interval in skew_model.intervals:
        flow = bl.heat(interval.terminal, bl.W_GRID, interval.t1 - interval.t0)
        assert np.max(np.abs(np.maximum.accumulate(flow) - interval.start)) < 1e-12
        assert np.array_equal(interval.mapping(interval.t1), np.interp(bl.W_GRID, bl.W_GRID, interval.terminal))


def test_simulated_paths_have_unit_mean(skew_paths):
    n = len(skew_paths)
    se = skew_paths.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(skew_paths.mean(axis=0) - 1) < 4 * se)


def test_simulated_paths_are_martingales(skew_paths):
    """No detectable drift of X_{i+1} - X_i conditional on X_i (20 decile tests)."""
    assert np.max(np.abs(conditional_drift_z(skew_paths))) < 4.5


def test_gauss_hermite_start_map_is_not_a_martingale():
    """Why the default is the exact convolution.

    With 24-node Gauss-Hermite quadrature the start map h differs from the
    exact expectation of the kinked terminal map, so E[X_{i+1} | X_i] != X_i
    by a clearly detectable amount, although every marginal is still matched.
    """
    maturities = np.array([0.1, 0.35, 0.85])  # a half-year last interval, as in the SPX example
    marginals = [bl.marginal_from_total_variance(skew_total_variance(bl.X_GRID, t), bl.X_GRID)
                 for t in maturities]
    for method, bound in (("gauss-hermite", lambda z: z > 6), ("exact", lambda z: z < 4.5)):
        model = bl.calibrate(marginals, maturities, method=method)
        paths = bl.simulate_market_dates(model, n=200_000, seed=11)
        assert bound(np.max(np.abs(conditional_drift_z(paths)))), method
