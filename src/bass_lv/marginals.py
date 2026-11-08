"""Target marginals in forward-normalized (mean-one) coordinates.

A marginal is stored by its quantile function on a fixed probability grid, which
is all the fixed point and the simulation need: ``ppf`` for the terminal map,
``cdf`` for the start map, and the quantiles themselves for call prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

#: Probability grid on which every marginal stores its quantiles.
U_GRID = np.linspace(1e-5, 1 - 1e-5, 4001)


@dataclass(frozen=True, eq=False)
class Marginal:
    """A law on (0, inf) with mean one, given by quantiles ``q`` on the grid ``u``."""

    q: np.ndarray
    u: np.ndarray = field(default_factory=lambda: U_GRID, repr=False)

    def ppf(self, p: np.ndarray | float) -> np.ndarray:
        """Quantile function ``F_mu^{-1}``."""
        return np.interp(p, self.u, self.q, left=self.q[0], right=self.q[-1])

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """Distribution function ``F_mu``."""
        return np.interp(x, self.q, self.u, left=0.0, right=1.0)

    def mean(self) -> float:
        """Mean, ``int_0^1 F_mu^{-1}(u) du``; equal to one by construction."""
        return float(np.trapezoid(self.q, self.u))

    def call(self, k: float) -> float:
        """Normalized call price ``E[(X - k)^+]`` from the quantiles."""
        return float(np.trapezoid(np.maximum(self.q - k, 0), self.u))


def marginal_from_quantiles(q: np.ndarray, u: np.ndarray = U_GRID) -> Marginal:
    """Make quantiles monotone and rescale them to mean one."""
    q = np.maximum.accumulate(np.asarray(q, float))
    q = q / np.trapezoid(q, u)
    return Marginal(q=q, u=u)


def marginal_from_total_variance(
    total_variance: np.ndarray, x_grid: np.ndarray, u: np.ndarray = U_GRID
) -> Marginal:
    """Breeden-Litzenberger marginal from a smile in total variance.

    ``total_variance`` is ``sigma_BS(k)^2 T`` on the log-moneyness grid
    ``x_grid = log(K/F)``. The CDF is ``F_mu(k) = 1 + dC/dk`` with ``C`` the
    forward-normalized Black call; it is clipped to [0, 1], made monotone and
    inverted numerically.
    """
    k = np.exp(x_grid)
    sqrt_w = np.sqrt(total_variance)
    d1 = -np.log(k) / sqrt_w + sqrt_w / 2
    call = norm.cdf(d1) - k * norm.cdf(d1 - sqrt_w)
    cdf = np.clip(1 + np.gradient(call, k), 0, 1)
    cdf = np.maximum.accumulate(cdf)
    cdf[0], cdf[-1] = 0.0, 1.0
    q = np.interp(u, cdf, k)
    return marginal_from_quantiles(q, u)


def lognormal_marginal(total_variance: float, u: np.ndarray = U_GRID) -> Marginal:
    """Mean-one lognormal law ``exp(-v/2 + sqrt(v) Z)`` with total variance ``v``."""
    q = np.exp(-total_variance / 2 + np.sqrt(total_variance) * norm.ppf(u))
    return marginal_from_quantiles(q, u)
