"""The heat semigroup K_t * h on the latent grid."""

import numpy as np
import pytest
from scipy.stats import norm

import bass_lv as bl

W = bl.W_GRID


def fine_convolution(values, at, dt, cdf=False):
    """Reference: trapezoid rule on a fine Gaussian grid against the same interpolant."""
    z = np.linspace(-10, 10, 80_001)
    density = norm.pdf(z)
    left, right = (0.0, 1.0) if cdf else (values[0], values[-1])
    sampled = np.interp(np.asarray(at)[:, None] + np.sqrt(dt) * z, W, values, left=left, right=right)
    return np.trapezoid(sampled * density, z, axis=1)


@pytest.fixture(scope="module")
def kinked_map():
    """Monotone, piecewise-linear map with jumps in slope, like a terminal map from a real smile."""
    rng = np.random.default_rng(0)
    steps = rng.exponential(1.0, W.size) * (1 + 5 * (rng.random(W.size) < 0.02))
    values = np.cumsum(steps)
    return values / values[-1]


@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("cdf", [False, True])
def test_exact_convolution_matches_fine_quadrature(kinked_map, dt, cdf):
    at = np.r_[W[::37], np.linspace(-7.5, 7.5, 41)]  # grid nodes, off-grid points, beyond the grid
    exact = bl.heat(kinked_map, at, dt, cdf=cdf)
    assert np.max(np.abs(exact - fine_convolution(kinked_map, at, dt, cdf))) < 1e-7


def test_gauss_hermite_is_accurate_only_for_smooth_maps(kinked_map):
    smooth = np.exp(0.2 * W)
    central = np.abs(W) <= 3
    for method in bl.METHODS:
        out = bl.heat(smooth, W, 0.5, method=method)
        assert np.max(np.abs(out / np.exp(0.2 * W + 0.01) - 1)[central]) < 1e-6
    at = W[central][::10]
    reference = fine_convolution(kinked_map, at, 0.5)
    exact_error = np.max(np.abs(bl.heat(kinked_map, at, 0.5) - reference))
    gh_error = np.max(np.abs(bl.heat(kinked_map, at, 0.5, method="gauss-hermite") - reference))
    assert exact_error < 1e-7 < 1e-4 < gh_error


def test_zero_time_is_interpolation(kinked_map):
    at = np.linspace(-8, 8, 101)
    assert np.array_equal(bl.heat(kinked_map, at, 0.0),
                          np.interp(at, W, kinked_map, left=kinked_map[0], right=kinked_map[-1]))


def test_semigroup_property():
    """K_0.2 * K_0.1 = K_0.3, up to re-interpolating the intermediate result on the grid."""
    values = norm.cdf(W / 0.7) + 0.1 * np.tanh(W)
    once = bl.heat(values, W, 0.3)
    twice = bl.heat(bl.heat(values, W, 0.1), W, 0.2)
    assert np.max(np.abs(once - twice)[np.abs(W) < 5]) < 1e-5


def test_heat_of_hat_closed_form():
    a, b, c = W[500], W[560], W[640]  # knots on grid nodes: the interpolant is the hat itself
    for dt in (0.05, 0.4):
        closed = bl.heat_hat(W, a, b, c, dt)
        assert np.max(np.abs(closed - bl.heat(bl.hat(W, a, b, c), W, dt))) < 1e-12


@pytest.mark.parametrize("cdf", [False, True])
def test_large_off_grid_input_matches_small_batches(kinked_map, cdf):
    """Off-grid points are evaluated in chunks; the result must not depend on the batch size."""
    at = np.linspace(-7.5, 7.5, 10_001) + 1e-3  # off the grid, more than two chunks
    whole = bl.heat(kinked_map, at.reshape(73, 137), 0.1, cdf=cdf)
    batches = np.concatenate([bl.heat(kinked_map, part, 0.1, cdf=cdf) for part in np.array_split(at, 7)])
    assert whole.shape == (73, 137)
    assert np.max(np.abs(whole.ravel() - batches)) < 1e-14


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        bl.heat(W, W, 0.1, method="simpson")
