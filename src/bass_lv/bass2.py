"""Bass2: the fixed-point operator with a general reference kernel (Section 6, Theorem 6.1).

Replacing the Brownian increment by an independent diffusion ``X`` (Eq. 16)
only changes the two kernels of the fixed point:

    A^X F = F_{mu_1} o K^{-X} * ( F_{mu_2}^{-1} o (K^X * F) ),    F = A^X F.   (Eq. 17)

``K^X`` pushes the latent CDF forward over the interval and ``K^{-X}`` pulls
the terminal map back. With Brownian kernels both are the heat kernel and Eq. (17)
reduces to Eq. (2).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .heat import DEFAULT_METHOD, W_GRID, heat
from .marginals import Marginal

Kernel = Callable[[np.ndarray], np.ndarray]


def bass2_operator(F: np.ndarray, mu1: Marginal, mu2: Marginal,
                   K_X: Kernel, K_minus_X: Kernel) -> np.ndarray:
    """Apply ``A^X`` of Eq. (17) to a latent CDF tabulated on the grid."""
    pre = K_X(F)                  # K^X * F
    terminal = mu2.ppf(pre)       # F_mu2^{-1} o (K^X * F)
    start = K_minus_X(terminal)   # K^{-X} * terminal
    return mu1.cdf(start)


def brownian_kernels(dt: float, grid: np.ndarray = W_GRID,
                     method: str = DEFAULT_METHOD) -> tuple[Kernel, Kernel]:
    """``(K^X, K^{-X})`` for a Brownian increment over ``dt``: both are the heat kernel."""
    def K_X(values: np.ndarray) -> np.ndarray:
        return heat(values, grid, dt, grid=grid, cdf=True, method=method)

    def K_minus_X(values: np.ndarray) -> np.ndarray:
        return heat(values, grid, dt, grid=grid, method=method)

    return K_X, K_minus_X
