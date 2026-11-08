"""Volatility Greeks by hat bumps of the terminal maps, and the vanilla hedge (Section 5).

The three steps of Section 5.1:

1. bump each terminal map ``f_i`` by a triangular hat ``h_i^l`` (Eq. 9);
2. differentiate the exotic price pathwise with a backward sweep over the
   market dates (Eqs. 10-11);
3. propagate the bump to the marginal CDFs by forward induction (Eqs. 12-15),
   integrate to vanilla call sensitivities, and solve for the vanilla portfolio
   whose bump sensitivities cancel the exotic's.

Paths are in forward-normalized units; ``forwards`` and ``discounts`` convert
to index points.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.stats import norm

from .calibration import BassLVModel
from .heat import heat

#: Normalized-asset grid on which marginal CDF sensitivities are tabulated.
ASSET_GRID = np.linspace(.25, 2.0, 1801)

Triangle = tuple[float, float, float]


def hat(w, a: float, b: float, c: float):
    """Triangular bump rising on ``[a, b]`` and falling on ``[b, c]`` (Eq. 9)."""
    return np.maximum(0, np.minimum((w - a) / (b - a), (c - w) / (c - b)))


def heat_hat(w, a: float, b: float, c: float, dt: float):
    """Closed form of ``K_dt * hat`` (displayed after Eq. 11)."""
    s = np.sqrt(dt)
    za, zb, zc = (a - w) / s, (b - w) / s, (c - w) / s
    left = ((w - a) * (norm.cdf(zb) - norm.cdf(za))
            + s * (norm.pdf(za) - norm.pdf(zb))) / (b - a)
    right = ((c - w) * (norm.cdf(zc) - norm.cdf(zb))
             + s * (norm.pdf(zc) - norm.pdf(zb))) / (c - b)
    return np.maximum(left + right, 0)


def _steps(model: BassLVModel) -> np.ndarray:
    return np.diff(np.r_[0, model.maturities])


def latent_cdfs_before(model: BassLVModel) -> list[np.ndarray]:
    """CDFs of ``W_{T_i-}``, the latent state just before each market date."""
    grid = model.grid
    cdfs = [norm.cdf(grid / np.sqrt(model.maturities[0]))]
    cdfs += [heat(iv.latent_cdf, grid, iv.t1 - iv.t0, grid=grid, cdf=True, method=model.method)
             for iv in model.pairs]
    return cdfs


def hat_triangles(
    model: BassLVModel, per_expiry: int = 3, coverage: tuple[float, float] = (.05, .95)
) -> list[list[Triangle]]:
    """Hat knots at quantiles of ``W_{T_i-}`` spanning its central mass (text below Eq. 9)."""
    knots = [np.interp(np.linspace(*coverage, per_expiry + 2), F, model.grid)
             for F in latent_cdfs_before(model)]
    return [[(k[n], k[n + 1], k[n + 2]) for n in range(per_expiry)] for k in knots]


def bump_terminal_map(model: BassLVModel, j: int, triangle: Triangle, eps: float) -> BassLVModel:
    """Model with ``f_j`` replaced by ``f_j + eps * hat`` (start map bumped consistently).

    The calibrated latent CDFs are left untouched. Used for bump-and-revalue checks.
    """
    a, b, c = triangle
    interval = model.intervals[j]
    dt = interval.t1 - interval.t0
    bumped = replace(
        interval,
        terminal=interval.terminal + eps * hat(model.grid, a, b, c),
        start=interval.start + eps * heat_hat(model.grid, a, b, c, dt),
    )
    intervals = list(model.intervals)
    intervals[j] = bumped
    return replace(model, intervals=intervals)


def simulate_latents(
    model: BassLVModel, n: int = 30_000, seed: int = 19
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate paths and keep the latent values around every market date.

    Returns ``(X, W_start, W_end)``, each of shape ``(n, m)``: ``W_start[:, i]``
    is the latent value at the start of interval ``i`` (after the Eq. 8 reset,
    and ``W_0 = f(0, .)^{-1}(1)`` for the first one) and ``W_end[:, i]`` the
    value just before ``T_i``.
    """
    rng = np.random.default_rng(seed)
    grid, dt, m = model.grid, _steps(model), len(model.maturities)
    W_start = np.zeros((n, m))
    W_end = np.zeros((n, m))
    path = np.zeros((n, m))
    first = model.intervals[0]
    W_start[:, 0] = np.interp(1.0, first.start, grid)  # keep S_0 = 1 under bumps
    W_end[:, 0] = W_start[:, 0] + np.sqrt(dt[0]) * rng.standard_normal(n)
    path[:, 0] = np.interp(W_end[:, 0], grid, first.terminal)
    for i in range(1, m):
        W_start[:, i] = model.intervals[i].reset(path[:, i - 1])  # Eq. (8)
        W_end[:, i] = W_start[:, i] + np.sqrt(dt[i]) * rng.standard_normal(n)
        path[:, i] = np.interp(W_end[:, i], grid, model.intervals[i].terminal)
    return path, W_start, W_end


def asian_payoff_gradient(paths: np.ndarray, forwards: np.ndarray, discount: float,
                          strike: float) -> np.ndarray:
    """``d payoff / d S_{T_i}`` for the discounted arithmetic Asian call; shape ``(n, m)``."""
    m = paths.shape[1]
    levels = paths * np.asarray(forwards)
    active = (levels.mean(axis=1) > strike).astype(float)
    return np.tile(discount * active[:, None] / m, (1, m))


def pathwise_adjoint(
    model: BassLVModel,
    payoff_gradient: np.ndarray,
    W_start: np.ndarray,
    W_end: np.ndarray,
    forwards: np.ndarray,
) -> np.ndarray:
    """Backward sweep of Eq. (10): total derivative of the payoff with respect to ``S_{T_i}``.

    Moving ``S_{T_i}`` moves ``W_{T_i}`` through the reset and hence every later
    fixing; along a path ``dS_{T_{i+1}} / dS_{T_i} = (F_{i+1} / F_i) g'_{i+1}(W_end) /
    h'_{i+1}(W_start)``.
    """
    grid = model.grid
    d_terminal = [np.gradient(iv.terminal, grid) for iv in model.intervals]
    d_start = [np.gradient(iv.start, grid) for iv in model.intervals]
    adjoint = np.array(payoff_gradient, dtype=float, copy=True)
    for i in range(adjoint.shape[1] - 2, -1, -1):
        numerator = np.interp(W_end[:, i + 1], grid, d_terminal[i + 1])
        denominator = np.interp(W_start[:, i + 1], grid, d_start[i + 1])
        transition = forwards[i + 1] / forwards[i] * numerator / np.maximum(denominator, 1e-8)
        adjoint[:, i] += adjoint[:, i + 1] * transition
    return adjoint


def exotic_hat_sensitivity(
    model: BassLVModel,
    j: int,
    triangle: Triangle,
    adjoint: np.ndarray,
    W_start: np.ndarray,
    W_end: np.ndarray,
    forwards: np.ndarray,
    with_se: bool = False,
) -> float | tuple[float, float]:
    """Price sensitivity to ``f_j -> f_j + eps * hat``, market-date term of Eq. (11).

    The bump moves ``S_{T_j}`` directly, by ``hat(W_end)``, and through the
    reset of ``W_start``, whose start map moves by ``K * hat``. For payoffs
    fixed only at market dates the within-interval sum of Eq. (11) is empty.
    With ``with_se=True`` the Monte Carlo standard error is returned as well.
    """
    a, b, c = triangle
    grid = model.grid
    interval = model.intervals[j]
    d_terminal = np.gradient(interval.terminal, grid)
    d_start = np.gradient(interval.start, grid)
    direct = hat(W_end[:, j], a, b, c)
    start_bump = heat_hat(W_start[:, j], a, b, c, _steps(model)[j])
    ratio = (np.interp(W_end[:, j], grid, d_terminal) /
             np.maximum(np.interp(W_start[:, j], grid, d_start), 1e-8))
    contributions = forwards[j] * (direct - start_bump * ratio) * adjoint[:, j]
    mean = float(np.mean(contributions))
    if with_se:
        return mean, float(contributions.std(ddof=1) / np.sqrt(len(contributions)))
    return mean


def implied_marginal_cdfs(model: BassLVModel, asset_grid: np.ndarray = ASSET_GRID) -> list[np.ndarray]:
    """CDFs of ``X_{T_i}`` implied by the model's maps, by forward induction (no Monte Carlo).

    ``F_{X_i}(x) = F_{W_{T_i-}}(g_i^{-1}(x))``, the reset gives
    ``F_{W_{T_i}}(w) = F_{X_i}(h_{i+1}(w))``, and the heat kernel carries it to
    ``W_{T_{i+1}-}``. For a calibrated model these are the target marginals;
    :func:`cdf_sensitivity` is the derivative of this map under a hat bump.
    """
    grid, dt = model.grid, _steps(model)
    W0 = np.interp(1.0, model.intervals[0].start, grid)
    before = norm.cdf((grid - W0) / np.sqrt(dt[0]))
    cdfs = []
    for i, interval in enumerate(model.intervals):
        cdf_x = np.interp(np.interp(asset_grid, interval.terminal, grid), grid, before)
        cdfs.append(cdf_x)
        if i + 1 < len(model.intervals):
            after = np.interp(model.intervals[i + 1].start, asset_grid, cdf_x)
            before = heat(after, grid, dt[i + 1], grid=grid, cdf=True, method=model.method)
    return cdfs


def cdf_sensitivity(
    model: BassLVModel, j: int, triangle: Triangle, asset_grid: np.ndarray = ASSET_GRID
) -> list[np.ndarray]:
    """Sensitivities of the marginal CDFs of ``X_{T_i}`` to a hat bump of ``f_j`` (Eqs. 12-15).

    Forward induction over the market dates: the bump changes the law of
    ``W_{T_i-}`` (from the ``W_0`` reset when ``j = 0``, from the start-map
    bump when ``i = j``), the inverse terminal map when ``i = j``, and is
    carried to the next interval by the reset and the heat kernel. Returns one
    array on ``asset_grid`` per market date.
    """
    a, b, c = triangle
    grid, dt, m = model.grid, _steps(model), len(model.maturities)
    terminal = [iv.terminal for iv in model.intervals]
    start = [iv.start for iv in model.intervals]
    d_terminal = [np.gradient(f, grid) for f in terminal]
    d_start = [np.gradient(f, grid) for f in start]
    before = latent_cdfs_before(model)
    h_grid = hat(grid, a, b, c)

    if j == 0:  # W_0 = f(0, .)^{-1}(S_0) moves; the Dirac initial law is shifted
        coefficient = heat_hat(0.0, a, b, c, dt[0]) / np.interp(0.0, grid, d_start[0])
        d_before = coefficient * norm.pdf(grid / np.sqrt(dt[0])) / np.sqrt(dt[0])
    else:
        d_before = np.zeros_like(grid)

    d_marginals = []
    for i in range(m):
        inverse = np.interp(asset_grid, terminal[i], grid)
        d_mu = np.interp(inverse, grid, d_before)
        if i == j:  # derivative of f_i^{-1}, Eq. (14)
            d_inverse = (-np.interp(inverse, grid, h_grid) /
                         np.maximum(np.interp(inverse, grid, d_terminal[i]), 1e-8))
            d_mu += np.interp(inverse, grid, np.gradient(before[i], grid)) * d_inverse
        d_marginals.append(d_mu)

        if i < m - 1:
            next_start = start[i + 1]
            d_start_cdf = np.interp(next_start, asset_grid, d_mu)
            if i + 1 == j:  # derivative of the next start map, Eq. (14)
                density = np.gradient(model.marginals[i].cdf(asset_grid), asset_grid)
                d_start_cdf += (np.interp(next_start, asset_grid, density) *
                                heat_hat(grid, a, b, c, dt[i + 1]))
            d_before = heat(d_start_cdf, grid, dt[i + 1], grid=grid, method=model.method)
    return d_marginals


def vanilla_sensitivities(
    model: BassLVModel,
    bumps: Sequence[tuple[int, Triangle]],
    instruments: Sequence[tuple[int, float]],
    forwards: np.ndarray,
    discounts: np.ndarray,
    asset_grid: np.ndarray = ASSET_GRID,
) -> np.ndarray:
    """Call-price sensitivities, rows = instruments ``(i, k)``, columns = bumps ``(j, hat)``.

    A call on ``S_{T_i}`` at normalized strike ``k`` is worth
    ``D_i F_i int_k^inf (1 - F_{X_i}(x)) dx``, so its sensitivity is
    ``-D_i F_i int_k^inf dF_{X_i}(x)``.
    """
    sens = np.zeros((len(instruments), len(bumps)))
    for column, (j, triangle) in enumerate(bumps):
        d_mu = cdf_sensitivity(model, j, triangle, asset_grid)
        for row, (i, k) in enumerate(instruments):
            normalized_call_sens = -_integral_from(k, asset_grid, d_mu[i])
            sens[row, column] = discounts[i] * forwards[i] * normalized_call_sens
    return sens


def _integral_from(k: float, x: np.ndarray, y: np.ndarray) -> float:
    """``int_k^{x[-1]} y dx`` by the trapezoid rule, with ``y`` interpolated at ``k``."""
    keep = x > k
    return float(np.trapezoid(np.r_[np.interp(k, x, y), y[keep]], np.r_[k, x[keep]]))


@dataclass
class HedgeResult:
    """Vanilla hedge neutralizing all bump sensitivities of an exotic."""

    weights: np.ndarray
    residual: np.ndarray
    rank: int
    condition_number: float


def solve_vega_hedge(vanilla_sens: np.ndarray, exotic_sens: np.ndarray,
                     rcond: float = 1e-9) -> HedgeResult:
    """Solve ``exotic + sum_k w_k vanilla_k`` = 0 for every bump (Section 5.1, step 3)."""
    target = -np.asarray(exotic_sens).ravel()
    weights, _, rank, _ = np.linalg.lstsq(vanilla_sens.T, target, rcond=rcond)
    residual = vanilla_sens.T @ weights - target
    return HedgeResult(weights, residual, int(rank), float(np.linalg.cond(vanilla_sens)))
