"""Bass-LV calibration: the classical first interval and the Theorem 2.1 fixed point.

Notation follows Conze and Henry-Labordere (2021). On ``[T_i, T_{i+1}]`` with
``Delta = T_{i+1} - T_i`` the unknown is the CDF ``F`` of the latent state
``W_{T_i}``. Theorem 2.1 characterizes it as the fixed point ``F = A F`` of

    A F = F_{mu_i} o K_Delta * ( F_{mu_{i+1}}^{-1} o (K_Delta * F) ),     (Eq. 2)

and the interval is then represented by its terminal map
``g = F_{mu_{i+1}}^{-1} o (K_Delta * F)`` through ``f(t, .) = K_{T_{i+1} - t} * g``
(Eq. 3). The asset is ``S_t = f(t, W_t)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from .heat import DEFAULT_METHOD, W_GRID, heat
from .marginals import Marginal


@dataclass(eq=False)
class Interval:
    """One calibrated interval ``[t0, t1]`` of the model.

    Attributes
    ----------
    terminal:
        ``g(w) = f(t1-, w)`` on the latent grid; maps ``W_{t1-}`` to ``S_{t1}``.
    start:
        ``h(w) = f(t0, w) = K_{t1 - t0} * g (w)``; its inverse is the Eq. (8) reset.
    latent_cdf:
        Fixed point ``F``, the CDF of ``W_{t0}``. ``None`` for the classical
        first interval, where ``W_0 = 0``.
    errors:
        Sup-norm change ``||F^{p+1} - F^p||`` at each fixed-point iteration.
    method:
        Convolution method used for ``K * .`` (see :mod:`bass_lv.heat`).
    converged:
        False if the fixed point stopped at ``max_iter`` before reaching
        ``tol``; always True for the classical first interval.
    """

    t0: float
    t1: float
    terminal: np.ndarray
    start: np.ndarray
    latent_cdf: np.ndarray | None = None
    errors: list[float] = field(default_factory=list)
    grid: np.ndarray = field(default_factory=lambda: W_GRID, repr=False)
    method: str = DEFAULT_METHOD
    converged: bool = True

    @property
    def iterations(self) -> int:
        return len(self.errors)

    def mapping(self, t: float, w: np.ndarray | None = None) -> np.ndarray:
        """``f(t, w) = K_{t1 - t} * g (w)`` (Eq. 3); ``w`` defaults to the latent grid."""
        return heat(self.terminal, self.grid if w is None else w, self.t1 - t,
                    grid=self.grid, method=self.method)

    def reset(self, spot: np.ndarray) -> np.ndarray:
        """Latent value after a market date, ``W_{t0} = h^{-1}(S_{t0})`` (Eq. 8)."""
        return np.interp(spot, self.start, self.grid, left=self.grid[0], right=self.grid[-1])

    def local_vol(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Local volatility at time ``t`` along the latent grid (Section 3.2).

        Returns ``(S, sigma)`` with ``S = f(t, w)`` and
        ``sigma(t, S) = d/dw log f(t, w)`` evaluated at ``w = f^{-1}(t, S)``.
        """
        remaining = self.t1 - t
        level = heat(self.terminal, self.grid, remaining, grid=self.grid, method=self.method)
        slope = heat(np.gradient(self.terminal, self.grid), self.grid, remaining,
                     grid=self.grid, method=self.method)
        return level, slope / level


@dataclass(eq=False)
class BassLVModel:
    """A Bass-LV model calibrated to marginals at ``maturities`` (all positive)."""

    maturities: np.ndarray
    marginals: list[Marginal]
    intervals: list[Interval]
    grid: np.ndarray = field(default_factory=lambda: W_GRID, repr=False)
    method: str = DEFAULT_METHOD

    @property
    def pairs(self) -> list[Interval]:
        """The intervals calibrated by the fixed point (all but the first)."""
        return self.intervals[1:]


def classical_interval(mu: Marginal, maturity: float, grid: np.ndarray = W_GRID,
                       method: str = DEFAULT_METHOD) -> Interval:
    """Section 1: ``f(T_1, w) = F_mu^{-1}(N(w / sqrt(T_1)))`` with ``W_0 = 0``."""
    terminal = mu.ppf(norm.cdf(grid / np.sqrt(maturity)))
    start = heat(terminal, grid, maturity, grid=grid, method=method)
    return Interval(0.0, float(maturity), terminal, start, None, [], grid, method)


def linearized_initial_cdf(
    mu1: Marginal, mu2: Marginal, dt: float, grid: np.ndarray = W_GRID
) -> np.ndarray:
    """Initial guess from the linearized fixed-point equation (Eq. 26).

    With ``G_i = F_{mu_i}^{-1}``,

        F_0^{-1}(u) = sqrt(Delta / 2) int_{1/2}^u sqrt( G_2'(y) / int_0^y (G_1 - G_2)(z) dz ) dy,

    so that ``F_0(0) = 1/2`` (the normalization of Remark 2.3). The tails are
    trimmed at ``2e-4`` because both the quantile derivative and the
    denominator are singular there.
    """
    p = np.linspace(2e-4, 1 - 2e-4, 4001)
    G1, G2 = mu1.ppf(p), mu2.ppf(p)
    denominator = cumulative_trapezoid(G1 - G2, p, initial=0)
    denominator += max(p[0] * (G1[0] - G2[0]), 0)
    positive = denominator[denominator > 0]
    floor = max(np.quantile(positive, .001) * 1e-4, 1e-14)
    integrand = np.sqrt(np.maximum(np.gradient(G2, p), 0) /
                        np.maximum(denominator, floor))
    primitive = cumulative_trapezoid(integrand, p, initial=0)
    latent_q = np.sqrt(dt / 2) * (primitive - np.interp(.5, p, primitive))
    return np.interp(grid, latent_q, p, left=0, right=1)


def recenter(cdf: np.ndarray, grid: np.ndarray = W_GRID) -> np.ndarray:
    """Translate a CDF so that ``F(0) = 1/2`` (Remark 2.3: ``A`` commutes with shifts)."""
    median = np.interp(.5, cdf, grid)
    return np.interp(grid + median, grid, cdf, left=0, right=1)


def bass_operator(F: np.ndarray, mu1: Marginal, mu2: Marginal, dt: float, *,
                  grid: np.ndarray = W_GRID, method: str = DEFAULT_METHOD) -> np.ndarray:
    """One application of ``A`` (Eq. 2) followed by the Remark 2.3 recentering."""
    pre = heat(F, grid, dt, grid=grid, cdf=True, method=method)        # K_Delta * F
    terminal = mu2.ppf(pre)                                            # F_mu2^{-1} o (K_Delta * F)
    start = heat(terminal, grid, dt, grid=grid, method=method)         # K_Delta * terminal
    return np.maximum.accumulate(recenter(mu1.cdf(start), grid))       # A F, centered


def calibrate_pair(
    mu1: Marginal,
    mu2: Marginal,
    t0: float,
    t1: float,
    *,
    grid: np.ndarray = W_GRID,
    tol: float = 1e-5,
    max_iter: int = 1000,
    initial: str | np.ndarray = "linearized",
    method: str = DEFAULT_METHOD,
) -> Interval:
    """Solve ``F = A F`` (Theorem 2.1) by Picard iteration, Section 3.4.1.

    ``initial`` is ``"linearized"`` (Eq. 26, the paper's recommendation),
    ``"gaussian"`` (``N(0, t0)``, exact in the Black-Scholes case), or an array.
    Iteration stops when the sup-norm change drops below ``tol``. If that does not
    happen within ``max_iter`` steps, a ``RuntimeWarning`` is issued and the
    returned interval has ``converged=False``.
    """
    dt = t1 - t0
    if isinstance(initial, str):
        if initial == "linearized":
            F = linearized_initial_cdf(mu1, mu2, dt, grid)
        elif initial == "gaussian":
            F = norm.cdf(grid / np.sqrt(t0))
        else:
            raise ValueError("initial must be 'linearized', 'gaussian' or an array")
    else:
        F = np.asarray(initial, float)

    errors: list[float] = []
    for _ in range(max_iter):
        new = bass_operator(F, mu1, mu2, dt, grid=grid, method=method)
        errors.append(float(np.max(np.abs(new - F))))
        F = new
        if errors[-1] < tol:
            break
    converged = bool(errors) and errors[-1] < tol
    if not converged:
        last = f"{errors[-1]:.1e}" if errors else "none"
        warnings.warn(f"fixed point on [{t0}, {t1}] did not converge in {max_iter} iterations "
                      f"(last step {last}, tol {tol:.1e})", RuntimeWarning, stacklevel=2)
    pre = heat(F, grid, dt, grid=grid, cdf=True, method=method)
    terminal = np.maximum.accumulate(mu2.ppf(pre))
    start = np.maximum.accumulate(heat(terminal, grid, dt, grid=grid, method=method))
    return Interval(float(t0), float(t1), terminal, start, F, errors, grid, method, converged)


def calibrate(
    marginals: Sequence[Marginal],
    maturities: Sequence[float],
    *,
    grid: np.ndarray = W_GRID,
    tol: float = 1e-5,
    max_iter: int = 1000,
    initial: str = "linearized",
    method: str = DEFAULT_METHOD,
) -> BassLVModel:
    """Calibrate every interval independently (Section 3.1, Eqs. 5-8).

    The first interval is the classical one-marginal Bass construction; each
    later interval solves its own two-marginal fixed point.
    """
    T = np.asarray(maturities, float)
    marginals = list(marginals)
    if len(T) != len(marginals):
        raise ValueError("need one marginal per maturity")
    if T[0] <= 0 or np.any(np.diff(T) <= 0):
        raise ValueError("maturities must be positive and increasing")
    intervals = [classical_interval(marginals[0], T[0], grid, method)]
    intervals += [
        calibrate_pair(marginals[i], marginals[i + 1], T[i], T[i + 1], grid=grid, tol=tol,
                       max_iter=max_iter, initial=initial, method=method)
        for i in range(len(T) - 1)
    ]
    return BassLVModel(T, marginals, intervals, grid, method)
