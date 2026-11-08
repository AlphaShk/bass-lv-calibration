"""Theorem 2.1: the fixed point F = A F on synthetic skewed marginals."""

import numpy as np
import pytest

import bass_lv as bl

from helpers import SKEW_MATURITIES


def test_fixed_point_converges(skew_model, skew_marginals):
    for i, pair in enumerate(skew_model.pairs):
        assert pair.converged and pair.iterations < 50
        assert pair.errors[-1] < 1e-5
        image = bl.bass_operator(pair.latent_cdf, skew_marginals[i], skew_marginals[i + 1],
                                 pair.t1 - pair.t0)
        assert np.max(np.abs(image - pair.latent_cdf)) < 2e-5
        assert abs(np.interp(0.0, bl.W_GRID, pair.latent_cdf) - 0.5) < 1e-3  # Remark 2.3
        assert np.all(np.diff(pair.latent_cdf) >= 0)


def test_iteration_contracts(skew_model):
    for pair in skew_model.pairs:
        errors = np.array(pair.errors)
        assert errors[-1] < 1e-2 * errors[0]
        assert np.all(errors[3:] <= errors[2:-1] * 1.0001)


@pytest.mark.parametrize("i", [0, 1])
def test_initializers_reach_the_same_fixed_point(skew_model, skew_marginals, i):
    linearized = skew_model.pairs[i]
    gaussian = bl.calibrate_pair(skew_marginals[i], skew_marginals[i + 1],
                                 SKEW_MATURITIES[i], SKEW_MATURITIES[i + 1], initial="gaussian")
    assert np.max(np.abs(gaussian.latent_cdf - linearized.latent_cdf)) < 1e-4


def test_non_convergence_is_reported(skew_marginals):
    with pytest.warns(RuntimeWarning, match="did not converge in 2 iterations"):
        pair = bl.calibrate_pair(skew_marginals[0], skew_marginals[1],
                                 SKEW_MATURITIES[0], SKEW_MATURITIES[1], max_iter=2)
    assert not pair.converged and pair.iterations == 2


def test_linearized_initializer_is_a_centred_cdf(skew_marginals):
    F0 = bl.linearized_initial_cdf(skew_marginals[0], skew_marginals[1], 0.25)
    assert np.all(np.diff(F0) >= 0) and F0[0] == 0 and F0[-1] == 1
    assert abs(np.interp(0.0, bl.W_GRID, F0) - 0.5) < 1e-3


def test_inputs_are_validated(skew_marginals):
    with pytest.raises(ValueError):
        bl.calibrate(skew_marginals, [0.1, 0.35])
    with pytest.raises(ValueError):
        bl.calibrate(skew_marginals, [0.1, 0.35, 0.35])
    with pytest.raises(ValueError):
        bl.calibrate_pair(skew_marginals[0], skew_marginals[1], 0.1, 0.35, initial="flat")
