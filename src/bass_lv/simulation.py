"""Exact large-step Monte Carlo (Section 3.4.2) and the payoffs priced on it.

Inside an interval the latent state moves by exact Gaussian increments and the
asset is read off the calibrated map, so there is no time-discretization error.
At each market date the latent state is reset with Eq. (8) to keep the asset
path continuous. Paths are returned in forward-normalized units
``X_i = S_{T_i} / F_i``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import BassLVModel
from .marginals import Marginal


def simulate_market_dates(model: BassLVModel, n: int = 60_000, seed: int = 7) -> np.ndarray:
    """Sample ``(X_{T_1}, ..., X_{T_m})`` at the calibrated dates; shape ``(n, m)``."""
    rng = np.random.default_rng(seed)
    T, grid = model.maturities, model.grid
    first = model.intervals[0]
    W = np.sqrt(T[0]) * rng.standard_normal(n)
    path = [np.interp(W, grid, first.terminal)]
    for i, interval in enumerate(model.pairs):
        W = interval.reset(path[-1])                       # Eq. (8)
        W += np.sqrt(T[i + 1] - T[i]) * rng.standard_normal(n)
        path.append(np.interp(W, grid, interval.terminal))
    return np.column_stack(path)


def repricing_table(
    paths: np.ndarray,
    marginals: list[Marginal],
    strikes: tuple[float, ...] = (0.9, 1.0, 1.1),
    labels: list | None = None,
) -> pd.DataFrame:
    """Monte Carlo versus target normalized call prices ``E[(X_i - k)^+]``."""
    labels = list(range(1, len(marginals) + 1)) if labels is None else labels
    n = len(paths)
    rows = []
    for i, label in enumerate(labels):
        for k in strikes:
            target = marginals[i].call(k)
            payoff = np.maximum(paths[:, i] - k, 0)
            mc = payoff.mean()
            se = payoff.std(ddof=1) / np.sqrt(n)
            rows.append((label, k, target, mc, mc - target, se))
    return pd.DataFrame(rows, columns=["expiry", "k", "target call", "MC call", "error", "MC s.e."])


def asian_call_payoffs(
    paths: np.ndarray, forwards: np.ndarray, discount: float, strike: float
) -> np.ndarray:
    """Discounted payoffs of ``max(mean_i S_{T_i} - K, 0)``, one per path."""
    levels = paths * np.asarray(forwards)
    return discount * np.maximum(levels.mean(axis=1) - strike, 0)


def asian_call(
    paths: np.ndarray, forwards: np.ndarray, discount: float, strike: float
) -> tuple[float, float]:
    """Arithmetic Asian call fixed at the market dates: price and MC standard error."""
    discounted = asian_call_payoffs(paths, forwards, discount, strike)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(len(discounted)))


def forward_start_calls(
    paths: np.ndarray, forwards: np.ndarray, i: int, strikes: np.ndarray
) -> np.ndarray:
    """Undiscounted forward-start calls ``E[(S_{T_{i+1}} / S_{T_i} - k)^+]`` (Section 4.2.1)."""
    levels = paths * np.asarray(forwards)
    ratio = levels[:, i + 1] / levels[:, i]
    return np.array([np.maximum(ratio - k, 0).mean() for k in strikes])


def martingale_diagnostic(paths: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Conditional drift ``E[X_{i+1} - X_i | X_i in bin]`` per interval and quantile bin.

    For a martingale every mean is zero up to Monte Carlo noise; ``z`` is the
    mean divided by its standard error. Unconditional means can look fine
    while these conditional means do not.
    """
    rows = []
    for i in range(paths.shape[1] - 1):
        edges = np.quantile(paths[:, i], np.linspace(0, 1, bins + 1)[1:-1])
        bucket = np.digitize(paths[:, i], edges)
        increment = paths[:, i + 1] - paths[:, i]
        for b in range(bins):
            x = increment[bucket == b]
            se = x.std(ddof=1) / np.sqrt(len(x))
            rows.append((i + 1, b + 1, x.mean(), se, x.mean() / se))
    return pd.DataFrame(rows, columns=["interval", "bin", "mean increment", "s.e.", "z"])
