"""Data preparation on a synthetic file in the vendor's Level-2 layout (no vendor data)."""

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bass_lv as bl
from bass_lv import data

from helpers import skew_total_variance

HEADER = ("UnderlyingSymbol,UnderlyingPrice,Exchange,OptionSymbol,Blank,Type,Expiration, DataDate,"
          "Strike,Last,Bid,Ask,Volume,OpenInterest,IV,Delta,Gamma,Theta,Vega,Alias")
SPOT = 2800.0
TERMS = {"09/20/2019": (0.999, 2801.0), "12/20/2019": (0.995, 2803.0), "06/19/2020": (0.985, 2806.0)}


def vendor_rows(symbol, expiry, discount, forward, date="08/15/2019 16:00"):
    T = (pd.Timestamp(expiry) - data.VALUATION_DATE).days / 365
    rows = []
    for K in np.arange(2000.0, 3601.0, 25.0):
        k = K / forward
        vol = np.sqrt(skew_total_variance(np.log(k), T) / T)
        call = discount * forward * bl.black_call(k, vol, T)
        put = call - discount * (forward - K)
        for kind, mid in (("call", call), ("put", put)):
            bid, ask = max(round(mid - 0.2, 2), 0.0), round(mid + 0.2, 2)
            code = f"{symbol}{expiry[8:]}{expiry[:2]}{expiry[3:5]}{kind[0].upper()}{int(K * 1000):08d}"
            rows.append(f"{symbol},{SPOT},*,{code},,{kind},{expiry},{date},{K:g},{mid:.2f},"
                        f"{bid},{ask},0,10,{vol:.4f},0,0,0,0,{code}")
    return rows


@pytest.fixture(scope="module")
def vendor_csv(tmp_path_factory):
    lines = [HEADER]
    for expiry, (discount, forward) in TERMS.items():
        lines += vendor_rows("SPX", expiry, discount, forward)
        lines += vendor_rows("SPXW", expiry, discount, forward)  # other root: excluded
    lines += vendor_rows("SPX", "10/18/2019", 0.998, 2802.0)    # other expiry: excluded
    lines += vendor_rows("SPX", "12/20/2019", 0.995, 2803.0, date="08/14/2019 16:00")[:4]  # other day
    lines += ["XYZ,50,*,XYZ190816C00045000,,call,08/16/2019,08/15/2019 16:00,45,0,4.9,5.3,0,0,0.5,1,0,0,0,x"]
    path = tmp_path_factory.mktemp("vendor") / data.SAMPLE_MEMBER
    path.write_text("\r\n".join(lines) + "\r\n")
    return path


@pytest.fixture(scope="module")
def vendor_zip(vendor_csv):
    path = vendor_csv.parent / data.SAMPLE_ARCHIVE
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(vendor_csv, data.SAMPLE_MEMBER)
    return path


def test_selection_keeps_one_root_day_and_three_expiries(vendor_csv, vendor_zip):
    chain, digest = data.read_vendor_chain(vendor_csv)
    chain_zip, digest_zip = data.read_vendor_chain(vendor_zip)
    pd.testing.assert_frame_equal(chain, chain_zip)
    assert digest == digest_zip and len(digest) == 64 and digest != data.SUBSET_SHA256
    assert set(chain.underlying) == {"SPX"} and set(chain.date) == {"08/15/2019"}
    assert sorted(chain.expiry.unique()) == list(data.EXPIRIES)
    assert len(chain) == 3 * 2 * 65


def test_otm_quotes_recover_forwards_and_follow_the_otm_rule(vendor_csv):
    chain, _ = data.read_vendor_chain(vendor_csv)
    quotes = data.build_otm_quotes(chain)
    for expiry, group in quotes.groupby("expiry"):
        discount, forward = TERMS[pd.Timestamp(expiry).strftime("%m/%d/%Y")]
        assert group.discount.iloc[0] == pytest.approx(discount, abs=2e-5)
        assert group.forward.iloc[0] == pytest.approx(forward, abs=0.05)
        assert set(group[group.strike < group.forward].type) == {"put"}
        assert set(group[group.strike >= group.forward].type) == {"call"}
        assert group.moneyness.between(0.65, 1.35).all()


def test_prepared_file_round_trips(vendor_csv, tmp_path):
    chain, _ = data.read_vendor_chain(vendor_csv)
    quotes = data.build_otm_quotes(chain)
    out = tmp_path / data.OTM_FILE
    quotes.to_csv(out, index=False)
    term, smiles, spot = data.load_otm_quotes(out)
    assert spot == SPOT and len(smiles) == 3 and list(term.columns)[:3] == ["T", "discount", "forward"]
    assert sum(len(s) for s in smiles) == len(quotes)
    reloaded = pd.concat(smiles, ignore_index=True)
    assert np.array_equal(reloaded.total_variance.to_numpy(), quotes.total_variance.to_numpy())


def test_prepare_script(vendor_zip, tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "quotes.csv"
    assert module.main([str(vendor_zip), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err  # synthetic input is not the published subset
    assert out.exists() and "wrote" in captured.out


def test_wrong_file_is_rejected(tmp_path):
    bad = tmp_path / "other.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        data.read_vendor_chain(bad)
