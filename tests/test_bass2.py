"""Theorem 6.1: the general-kernel operator reduces to Eq. (2) for Brownian kernels."""

import numpy as np
import pytest

import bass_lv as bl


@pytest.mark.parametrize("method", bl.METHODS)
def test_brownian_kernels_give_the_theorem_2_1_operator(skew_model, skew_marginals, method):
    pair = skew_model.pairs[0]
    dt = pair.t1 - pair.t0
    K_X, K_minus_X = bl.brownian_kernels(dt, method=method)
    general = bl.bass2_operator(pair.latent_cdf, skew_marginals[0], skew_marginals[1], K_X, K_minus_X)
    direct = bl.bass_operator(pair.latent_cdf, skew_marginals[0], skew_marginals[1], dt, method=method)
    assert np.array_equal(np.maximum.accumulate(bl.recenter(general)), direct)


def test_calibrated_cdf_is_a_fixed_point_of_the_general_operator(skew_model, skew_marginals):
    for i, pair in enumerate(skew_model.pairs):
        K_X, K_minus_X = bl.brownian_kernels(pair.t1 - pair.t0)
        image = bl.recenter(bl.bass2_operator(pair.latent_cdf, skew_marginals[i],
                                              skew_marginals[i + 1], K_X, K_minus_X))
        assert np.max(np.abs(image - pair.latent_cdf)) < 2e-5
