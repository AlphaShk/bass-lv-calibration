"""Market preprocessing on a synthetic option chain."""

import numpy as np
import pandas as pd
import pytest

import bass_lv as bl

from helpers import skew_total_variance

VALUATION = pd.Timestamp("2024-01-02")
SPOT = 100.0
STRIKES = np.arange(60, 145, 2.5)


def synthetic_chain(discount=0.98, forward=101.3, days=180, strikes=STRIKES):
    """Calls and puts priced off a skewed smile, with a small symmetric spread."""
    expiry = VALUATION + pd.Timedelta(days=days)
    T = days / 365
    k = strikes / forward
    vol = np.sqrt(skew_total_variance(np.log(k), T) / T)
    call = discount * forward * bl.black_call(k, vol, T)
    put = call - discount * (forward - strikes)
    rows = []
    for K, c, p in zip(strikes, call, put, strict=True):
        for kind, mid in (("call", c), ("put", p)):
            rows.append({"expiry": expiry, "type": kind, "strike": K,
                         "bid": mid - 0.05, "ask": mid + 0.05, "mid": mid})
    return pd.DataFrame(rows), T, vol


def test_black_call_and_implied_vol_round_trip():
    for k in (0.9, 1.0, 1.15):
        for sigma in (0.1, 0.2, 0.8):
            assert abs(bl.implied_vol(bl.black_call(k, sigma, 0.5), k, 0.5) - sigma) < 1e-10


def test_put_call_parity_recovers_forward_and_discount():
    chain, T, vol = synthetic_chain()
    term = bl.infer_forward_discount(chain, VALUATION, SPOT)
    row = term.iloc[0]
    assert abs(row.discount - 0.98) < 1e-12 and abs(row.forward - 101.3) < 1e-9
    assert row["T"] == pytest.approx(T)
    assert row.atm_strike == 102.5  # listed strike nearest the forward
    k_atm = 102.5 / 101.3
    assert row.atm_iv == pytest.approx(np.sqrt(skew_total_variance(np.log(k_atm), T) / T), abs=1e-9)


def test_otm_selection_applies_every_filter():
    chain, T, _ = synthetic_chain()
    F, D = 101.3, 0.98
    chain.loc[(chain.strike == 90) & (chain.type == "put"), "bid"] = 0.0                  # zero bid
    chain.loc[(chain.strike == 110) & (chain.type == "call"), "ask"] += 50.0              # wide spread
    chain["mid"] = (chain.bid + chain.ask) / 2
    otm = bl.select_otm_quotes(chain, F, D, T)
    assert 90 not in otm.strike.values and 110 not in otm.strike.values
    assert otm.strike.min() > 0.65 * F and otm.strike.max() < 1.35 * F
    assert set(otm[otm.strike < F].type) == {"put"} and set(otm[otm.strike >= F].type) == {"call"}
    assert np.all(np.diff(otm.strike) > 0)
    expected = skew_total_variance(otm.log_moneyness, T)
    assert np.max(np.abs(otm.total_variance - expected)) < 1e-10


def test_total_variance_is_made_calendar_monotone():
    x = np.linspace(-0.3, 0.3, 25)
    short = pd.DataFrame({"log_moneyness": x, "total_variance": 0.02 + 0.2 * x**2})
    long = pd.DataFrame({"log_moneyness": x, "total_variance": 0.03 + 0.01 * x})  # crosses below
    fitted = bl.smooth_total_variance([short, long])
    assert fitted.shape == (2, bl.X_GRID.size)
    assert np.all(fitted[1] >= fitted[0]) and np.all(fitted > 0)
