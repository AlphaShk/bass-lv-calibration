"""Market-data preprocessing: forwards, OTM smiles and total-variance fits.

None of this is part of the Bass-LV construction itself. It turns an option
chain with real rates and dividends into the mean-one marginals the paper takes
as input (its footnote 1 reduces the problem to a positive martingale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

#: Log-moneyness grid on which smiles are smoothed and marginals are built.
X_GRID = np.linspace(np.log(0.45), np.log(1.80), 1001)


def black_call(k, sigma, T):
    """Forward-normalized Black call ``C / (D F)`` at normalized strike ``k = K / F``."""
    s = sigma * np.sqrt(T)
    d1 = -np.log(k) / s + s / 2
    return norm.cdf(d1) - k * norm.cdf(d1 - s)


def implied_vol(c: float, k: float, T: float, lo: float = 1e-4, hi: float = 3.0) -> float:
    """Black implied volatility of a normalized call price (Brent root finding)."""
    return brentq(lambda sigma: black_call(k, sigma, T) - c, lo, hi)


def infer_forward_discount(
    chain: pd.DataFrame,
    valuation_date: pd.Timestamp,
    spot: float,
    band: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """Discount factor, forward and ATM volatility per expiry from put-call parity.

    For each expiry, regress call-minus-put midpoints on strike,
    ``C(K) - P(K) = D (F - K)``, using strikes inside ``band * spot``. The ATM
    volatility is the implied volatility of the call midpoint at the listed
    strike nearest to the forward.

    ``chain`` needs the columns ``expiry`` (datetime), ``type`` ("call"/"put"),
    ``strike`` and ``mid``.
    """
    rows = {}
    for expiry, group in chain.groupby("expiry", sort=True):
        pairs = group.pivot(index="strike", columns="type", values="mid").dropna()
        near = pairs.loc[(pairs.index > band[0] * spot) & (pairs.index < band[1] * spot)]
        slope, intercept = np.polyfit(near.index, near.call - near.put, 1)
        D, F = -slope, intercept / (-slope)
        K = pairs.index[np.argmin(np.abs(pairs.index - F))]
        c = pairs.loc[K, "call"] / (D * F)
        k = K / F
        T = (pd.Timestamp(expiry) - valuation_date).days / 365
        rows[pd.Timestamp(expiry)] = {
            "T": T, "discount": D, "forward": F, "atm_strike": K, "atm_iv": implied_vol(c, k, T),
        }
    term = pd.DataFrame.from_dict(rows, orient="index")
    term.index.name = "expiry"
    return term


def select_otm_quotes(
    quotes: pd.DataFrame,
    F: float,
    D: float,
    T: float,
    moneyness: tuple[float, float] = (0.65, 1.35),
    min_spread_cap: float = 10.0,
    relative_spread_cap: float = 0.5,
) -> pd.DataFrame:
    """Clean out-of-the-money quotes of one expiry and convert them to total variance.

    Filters, in order:

    1. puts for strikes below the forward, calls at or above it;
    2. normalized strike ``K / F`` strictly inside ``moneyness``;
    3. positive bid and ``ask >= bid``;
    4. bid-ask spread at most ``max(min_spread_cap, relative_spread_cap * mid)``;
    5. normalized call price (puts via parity) strictly between intrinsic value and 1;
    6. a Black implied volatility exists in [1e-4, 3].

    Returns one row per kept quote with the columns ``strike, type, bid, ask,
    mid, moneyness, log_moneyness, implied_vol, total_variance``, sorted by strike.
    """
    book = quotes.set_index(["strike", "type"])
    points = []
    for K in sorted(quotes.strike.unique()):
        k = K / F
        option_type = "put" if K < F else "call"
        if not (moneyness[0] < k < moneyness[1]) or (K, option_type) not in book.index:
            continue
        quote = book.loc[(K, option_type)]
        if isinstance(quote, pd.DataFrame):
            quote = quote.iloc[0]
        if quote.bid <= 0 or quote.ask < quote.bid:
            continue
        if quote.ask - quote.bid > max(min_spread_cap, relative_spread_cap * quote.mid):
            continue
        c = quote.mid / (D * F)
        if option_type == "put":
            c = c + 1 - k  # normalized call-put parity
        intrinsic = max(1 - k, 0)
        if not (intrinsic < c < 1):
            continue
        try:
            iv = implied_vol(c, k, T)
        except ValueError:
            continue
        points.append((K, option_type, quote.bid, quote.ask, quote.mid,
                       k, np.log(k), iv, iv * iv * T))
    return pd.DataFrame(points, columns=[
        "strike", "type", "bid", "ask", "mid",
        "moneyness", "log_moneyness", "implied_vol", "total_variance",
    ])


def smooth_total_variance(
    smiles: list[pd.DataFrame],
    x_grid: np.ndarray = X_GRID,
    smoothing: float = 2e-5,
    floor: float = 1e-6,
) -> np.ndarray:
    """Smooth each smile in log-moneyness and impose calendar monotonicity.

    Each expiry gets a cubic smoothing spline of total variance against
    log-moneyness (smoothing factor ``smoothing * n_quotes``), held flat beyond
    the last quote. Total variance is then made nondecreasing across maturities,
    a necessary condition for the marginals to be in convex order.

    Returns an array of shape ``(n_expiries, len(x_grid))``.
    """
    fitted = []
    for points in smiles:
        spline = UnivariateSpline(points.log_moneyness, points.total_variance,
                                  s=smoothing * len(points), ext=3)
        fitted.append(np.maximum(spline(x_grid), floor))
    return np.maximum.accumulate(np.asarray(fitted), axis=0)
