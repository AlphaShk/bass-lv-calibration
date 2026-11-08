"""Heat semigroup on the latent Brownian coordinate.

``K_t * h (w) = E[h(w + sqrt(t) Z)]`` with ``Z ~ N(0, 1)`` (paper, Section 1).
Functions are tabulated on a latent grid and read between nodes by linear
interpolation, held constant beyond the grid (CDFs: extended by 0 and 1). This
is exactly how the Monte Carlo evaluates the maps, so the convolution should be
exact for that interpolant. Two methods are available:

``"exact"`` (default)
    Closed-form convolution of the piecewise-linear interpolant with the
    Gaussian density. Exact up to rounding, whatever the smoothness of ``h``.
``"gauss-hermite"``
    24-node Gauss-Hermite quadrature: the paper suggests Gauss-Hermite
    quadrature for these convolutions (Section 3.4.1), and the first
    implementation used 24 nodes. Accurate for smooth ``h`` but not for the
    kinked maps that come out of a spline-smoothed smile (see the README,
    "Numerical note").

Off the grid, the exact method builds ``len(at) x len(grid)`` weight matrices,
so ``at`` is processed in chunks of ``_CHUNK`` points (about 40 MB per
temporary array). On the grid the weights are cached.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

#: Latent grid for W. Wide enough that the boundary is irrelevant for
#: maturities up to about one year.
W_GRID = np.linspace(-7.0, 7.0, 1201)

#: Number of Gauss-Hermite nodes used by the ``"gauss-hermite"`` method.
GH_ORDER = 24

METHODS = ("exact", "gauss-hermite")
DEFAULT_METHOD = "exact"

#: Off-grid points evaluated per block by the exact method, which bounds the
#: dense ``_CHUNK x len(grid)`` temporaries (4096 x 1201 doubles, about 40 MB).
_CHUNK = 4096


def gauss_hermite(order: int = GH_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Nodes ``x_j`` and weights ``a_j`` with ``E[g(Z)] ~ sum_j a_j g(sqrt(2) x_j)``."""
    nodes, weights = hermgauss(order)
    return nodes, weights / np.sqrt(np.pi)


_NODES, _WEIGHTS = gauss_hermite(GH_ORDER)


def _ramp_weights(at: np.ndarray, dt: float, grid: np.ndarray) -> np.ndarray:
    """``B[i, k] = E[ramp_k(at_i + sqrt(dt) Z)]`` for the unit ramps between grid nodes.

    The interpolant is ``v(y) = v_0 + sum_k (v_{k+1} - v_k) ramp_k(y)`` with
    ``ramp_k = clip((y - x_k) / (x_{k+1} - x_k), 0, 1)``, and
    ``E[(a + sqrt(dt) Z)^+] = s (a/s Phi(a/s) + phi(a/s))`` with ``s = sqrt(dt)``.
    """
    s = np.sqrt(dt)
    a = (at[:, None] - grid[None, :]) / s
    call = s * (a * norm.cdf(a) + norm.pdf(a))
    return (call[:, :-1] - call[:, 1:]) / np.diff(grid)


@lru_cache(maxsize=8)
def _grid_weights(dt: float, grid_bytes: bytes) -> np.ndarray:
    grid = np.frombuffer(grid_bytes)
    return _ramp_weights(grid, dt, grid)


def _exact(values: np.ndarray, at: np.ndarray, dt: float, grid: np.ndarray, cdf: bool) -> np.ndarray:
    values = np.asarray(values, float)
    flat = at.ravel()
    increments = np.diff(values)
    if at.shape == grid.shape and np.array_equal(at, grid):
        weights = _grid_weights(float(dt), np.ascontiguousarray(grid, float).tobytes())
        out = values[0] + weights @ increments
    else:  # dense weights, built for at most _CHUNK points at a time
        out = np.empty(flat.shape)
        for lo in range(0, flat.size, _CHUNK):
            part = slice(lo, lo + _CHUNK)
            out[part] = values[0] + _ramp_weights(flat[part], dt, grid) @ increments
    if cdf:  # extension by 0 on the left and 1 on the right instead of the end values
        s = np.sqrt(dt)
        out = (out - values[0] * norm.cdf((grid[0] - flat) / s)
               + (1 - values[-1]) * norm.cdf((flat - grid[-1]) / s))
    return out.reshape(at.shape)


def heat(
    values: np.ndarray,
    at: np.ndarray | float,
    dt: float,
    *,
    grid: np.ndarray = W_GRID,
    cdf: bool = False,
    method: str = DEFAULT_METHOD,
) -> np.ndarray:
    """Evaluate ``K_dt * values`` at the points ``at``.

    Parameters
    ----------
    values:
        Function tabulated on ``grid``.
    at:
        Points where the convolution is evaluated (any shape). Off the grid the
        exact method costs ``O(len(at) * len(grid))`` time; memory is bounded by
        evaluating ``_CHUNK`` points at a time.
    dt:
        Variance of the Gaussian increment (a time step, not its square root).
    cdf:
        If True, ``values`` is a CDF and is extended by 0 and 1 outside the
        grid; otherwise it is held constant at its boundary values.
    method:
        ``"exact"`` or ``"gauss-hermite"`` (see the module docstring).
    """
    at = np.asarray(at)
    if method == "exact":
        if dt <= 0:
            left, right = (0.0, 1.0) if cdf else (values[0], values[-1])
            return np.interp(at, grid, values, left=left, right=right)
        return _exact(values, at, dt, grid, cdf)
    if method != "gauss-hermite":
        raise ValueError(f"method must be one of {METHODS}")
    z = at[..., None] + np.sqrt(2 * max(dt, 0)) * _NODES
    left, right = (0.0, 1.0) if cdf else (values[0], values[-1])
    return (np.interp(z, grid, values, left=left, right=right) * _WEIGHTS).sum(axis=-1)
