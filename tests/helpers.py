"""Constants and helpers shared by the tests (synthetic inputs only)."""

import numpy as np

SIGMA = 0.2
FLAT_MATURITIES = np.array([0.25, 0.5, 1.0])
SKEW_MATURITIES = np.array([0.1, 0.35, 0.6])


def skew_total_variance(x, T, atm=0.2, beta=0.1, rho=-0.7, s=0.1):
    """SVI-type smile in total variance; linear in T, hence calendar-monotone."""
    return T * (atm**2 + beta * (rho * x + np.sqrt(x**2 + s**2) - s))


def call_from_cdf(cdf, grid, k):
    """``int_k^inf (1 - F(x)) dx`` with the integrand interpolated at ``k``."""
    keep = grid > k
    x = np.r_[k, grid[keep]]
    y = 1 - np.r_[np.interp(k, grid, cdf), cdf[keep]]
    return np.trapezoid(y, x)
