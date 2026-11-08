"""Bass local-volatility model with multiple marginals.

A replication of A. Conze and P. Henry-Labordere, "Bass Construction with
Multi-Marginals: Lightspeed Computation in a New Local Volatility Model"
(SSRN 3853085, 2021).
"""

from .bass2 import bass2_operator, brownian_kernels
from .calibration import (
    BassLVModel,
    Interval,
    bass_operator,
    calibrate,
    calibrate_pair,
    classical_interval,
    linearized_initial_cdf,
    recenter,
)
from .greeks import (
    ASSET_GRID,
    HedgeResult,
    asian_payoff_gradient,
    bump_terminal_map,
    cdf_sensitivity,
    exotic_hat_sensitivity,
    hat,
    hat_triangles,
    heat_hat,
    implied_marginal_cdfs,
    latent_cdfs_before,
    pathwise_adjoint,
    simulate_latents,
    solve_vega_hedge,
    vanilla_sensitivities,
)
from .heat import DEFAULT_METHOD, GH_ORDER, METHODS, W_GRID, gauss_hermite, heat
from .marginals import (
    U_GRID,
    Marginal,
    lognormal_marginal,
    marginal_from_quantiles,
    marginal_from_total_variance,
)
from .market import (
    X_GRID,
    black_call,
    implied_vol,
    infer_forward_discount,
    select_otm_quotes,
    smooth_total_variance,
)
from .simulation import (
    asian_call,
    asian_call_payoffs,
    forward_start_calls,
    martingale_diagnostic,
    repricing_table,
    simulate_market_dates,
)

__version__ = "0.1.0"

__all__ = [
    "ASSET_GRID", "DEFAULT_METHOD", "GH_ORDER", "METHODS", "U_GRID", "W_GRID", "X_GRID",
    "BassLVModel", "HedgeResult", "Interval", "Marginal",
    "asian_call", "asian_call_payoffs", "asian_payoff_gradient", "bass2_operator", "bass_operator",
    "black_call", "brownian_kernels", "bump_terminal_map", "calibrate", "calibrate_pair",
    "cdf_sensitivity", "classical_interval", "exotic_hat_sensitivity",
    "forward_start_calls", "gauss_hermite", "hat", "hat_triangles", "heat", "heat_hat",
    "implied_marginal_cdfs", "implied_vol", "infer_forward_discount", "latent_cdfs_before",
    "linearized_initial_cdf", "lognormal_marginal", "marginal_from_quantiles",
    "marginal_from_total_variance", "martingale_diagnostic", "pathwise_adjoint", "recenter",
    "repricing_table",
    "select_otm_quotes", "simulate_latents", "simulate_market_dates",
    "smooth_total_variance", "solve_vega_hedge", "vanilla_sensitivities",
]
