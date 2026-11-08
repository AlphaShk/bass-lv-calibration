"""Synthetic inputs shared by the tests. No market data is used anywhere in the suite."""

import pytest

import bass_lv as bl

from helpers import FLAT_MATURITIES, SIGMA, SKEW_MATURITIES, skew_total_variance


@pytest.fixture(scope="session")
def flat_marginals():
    return [bl.lognormal_marginal(SIGMA**2 * t) for t in FLAT_MATURITIES]


@pytest.fixture(scope="session")
def flat_model(flat_marginals):
    return bl.calibrate(flat_marginals, FLAT_MATURITIES)


@pytest.fixture(scope="session")
def skew_marginals():
    return [bl.marginal_from_total_variance(skew_total_variance(bl.X_GRID, t), bl.X_GRID)
            for t in SKEW_MATURITIES]


@pytest.fixture(scope="session")
def skew_model(skew_marginals):
    return bl.calibrate(skew_marginals, SKEW_MATURITIES)


@pytest.fixture(scope="session")
def skew_paths(skew_model):
    return bl.simulate_market_dates(skew_model, n=200_000, seed=11)
