"""Section 5: hat-bump Greeks checked against bump-and-revalue, and the vanilla hedge."""

import numpy as np
import pytest

import bass_lv as bl

from helpers import call_from_cdf

STRIKES = (0.9, 1.0, 1.1)
INSTRUMENTS = [(i, k) for i in range(3) for k in STRIKES]
ONES = np.ones(3)


@pytest.fixture(scope="module")
def bumps(skew_model):
    triangles = bl.hat_triangles(skew_model, per_expiry=3)
    return [(j, t) for j in range(3) for t in triangles[j]]


def implied_calls(model):
    cdfs = bl.implied_marginal_cdfs(model)
    return np.array([call_from_cdf(cdfs[i], bl.ASSET_GRID, k) for i, k in INSTRUMENTS])


def test_forward_induction_recovers_the_target_marginals(skew_model, skew_marginals):
    targets = np.array([skew_marginals[i].call(k) for i, k in INSTRUMENTS])
    assert np.max(np.abs(implied_calls(skew_model) - targets)) < 1e-4


def test_hat_knots_cover_the_latent_law(skew_model):
    triangles = bl.hat_triangles(skew_model, per_expiry=4)
    for F, row in zip(bl.latent_cdfs_before(skew_model), triangles, strict=True):
        assert len(row) == 4
        knots = np.r_[row[0][0], [t[1] for t in row], row[-1][2]]
        assert np.all(np.diff(knots) > 0)
        mass = np.interp(knots[[0, -1]], bl.W_GRID, F)
        assert mass[0] == pytest.approx(0.05, abs=1e-3) and mass[1] == pytest.approx(0.95, abs=1e-3)


def test_vanilla_sensitivities_match_finite_differences(skew_model, bumps):
    """Eqs. (12)-(15) against central differences of the forward induction."""
    semi = bl.vanilla_sensitivities(skew_model, bumps, INSTRUMENTS, ONES, ONES)
    eps = 1e-4
    for column, (j, triangle) in enumerate(bumps):
        up = implied_calls(bl.bump_terminal_map(skew_model, j, triangle, eps))
        down = implied_calls(bl.bump_terminal_map(skew_model, j, triangle, -eps))
        fd = (up - down) / (2 * eps)
        assert np.max(np.abs(semi[:, column] - fd)) < 0.03 * np.max(np.abs(fd))


def test_asian_sensitivities_match_bump_and_revalue(skew_model, bumps):
    """Eqs. (10)-(11) against Monte Carlo bump-and-revalue with common random numbers."""
    n, seed, eps = 100_000, 8, 1e-3
    X, W_start, W_end = bl.simulate_latents(skew_model, n=n, seed=seed)
    gradient = bl.asian_payoff_gradient(X, ONES, 1.0, 1.0)
    adjoint = bl.pathwise_adjoint(skew_model, gradient, W_start, W_end, ONES)

    def price(model):
        return bl.asian_call(bl.simulate_latents(model, n=n, seed=seed)[0], ONES, 1.0, 1.0)[0]

    pathwise, fd = [], []
    for j, triangle in bumps:
        pathwise.append(bl.exotic_hat_sensitivity(skew_model, j, triangle, adjoint, W_start, W_end, ONES))
        fd.append((price(bl.bump_terminal_map(skew_model, j, triangle, eps))
                   - price(bl.bump_terminal_map(skew_model, j, triangle, -eps))) / (2 * eps))
    pathwise, fd = np.array(pathwise), np.array(fd)
    assert np.max(np.abs(pathwise - fd)) < 0.01 * np.max(np.abs(fd))


def test_vega_hedge_neutralizes_every_bump(skew_model, bumps):
    X, W_start, W_end = bl.simulate_latents(skew_model, n=20_000, seed=1)
    adjoint = bl.pathwise_adjoint(skew_model, bl.asian_payoff_gradient(X, ONES, 1.0, 1.0),
                                  W_start, W_end, ONES)
    exotic = np.array([bl.exotic_hat_sensitivity(skew_model, j, t, adjoint, W_start, W_end, ONES)
                       for j, t in bumps])
    vanilla = bl.vanilla_sensitivities(skew_model, bumps, INSTRUMENTS, ONES, ONES)
    hedge = bl.solve_vega_hedge(vanilla, exotic)
    assert hedge.rank == 9 and np.isfinite(hedge.condition_number)
    assert np.max(np.abs(hedge.residual)) < 1e-8 * np.max(np.abs(exotic))
