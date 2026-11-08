"""Exact large-step Monte Carlo: vanilla repricing and payoff helpers."""

import numpy as np

import bass_lv as bl


def test_monte_carlo_reprices_vanilla_calls(skew_model, skew_marginals):
    paths = bl.simulate_market_dates(skew_model, n=60_000, seed=7)
    table = bl.repricing_table(paths, skew_marginals)
    assert len(table) == 9
    assert np.all(np.abs(table["error"]) < 4 * table["MC s.e."])
    assert np.all(np.abs(table["error"]) < 1e-3)


def test_simulation_is_reproducible(skew_model):
    a = bl.simulate_market_dates(skew_model, n=1000, seed=5)
    b = bl.simulate_market_dates(skew_model, n=1000, seed=5)
    assert np.array_equal(a, b) and a.shape == (1000, 3)


def test_asian_call_is_below_the_average_of_calls(skew_paths):
    forwards, strike, discount = np.array([100.0, 100.5, 101.0]), 100.0, 0.98
    asian, se = bl.asian_call(skew_paths, forwards, discount, strike)
    calls = discount * np.maximum(skew_paths * forwards - strike, 0).mean(axis=0)
    assert 0 < asian <= calls.mean() and se > 0


def test_forward_start_calls_are_decreasing_and_convex_in_strike(skew_paths):
    strikes = np.linspace(0.8, 1.2, 21)
    prices = bl.forward_start_calls(skew_paths, np.ones(3), 0, strikes)
    assert np.all(np.diff(prices) < 0)
    assert np.all(np.diff(prices, 2) > -1e-12)
