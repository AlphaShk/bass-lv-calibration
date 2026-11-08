"""The SPX input: HistoricalOptionData's free Level-2 end-of-day sample for 15 Aug 2019.

The vendor file is not redistributed with this repository. Download
``Sample_L2_20190815.zip`` (End-of-Day Options Data, Level 2, daily sample) from
https://historicaloptiondata.com/sample-files/ and run
``python scripts/prepare_data.py path/to/Sample_L2_20190815.zip``.

Pipeline:

1. keep the standard SPX root (not SPXW) quoted at the 16:00 close of 15 Aug 2019
   for the expiries 20 Sep 2019, 20 Dec 2019 and 19 Jun 2020 (1,026 calls and puts);
2. infer discount factor and forward per expiry from put-call parity;
3. keep cleaned out-of-the-money quotes (filters in
   :func:`bass_lv.market.select_otm_quotes`): 246, 65 and 67 quotes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .market import infer_forward_discount, select_otm_quotes

SAMPLE_PAGE = "https://historicaloptiondata.com/sample-files/"
SAMPLE_ARCHIVE = "Sample_L2_20190815.zip"
SAMPLE_MEMBER = "options_20190815.csv"
OTM_FILE = "spx_otm_quotes_2019-08-15.csv"

UNDERLYING = "SPX"
VALUATION_DATE = pd.Timestamp("2019-08-15")
EXPIRIES = (pd.Timestamp("2019-09-20"), pd.Timestamp("2019-12-20"), pd.Timestamp("2020-06-19"))

#: SHA-256 of the selected raw vendor lines (each terminated by "\n", in file
#: order) that produced the results in the README.
SUBSET_SHA256 = "9f44759c674e8769677231e653c26d61537e373bae7a4a0b0fd897928c1ba88f"

_VENDOR_COLUMNS = {
    "UnderlyingSymbol": "underlying",
    "UnderlyingPrice": "spot",
    "Type": "type",
    "Expiration": "expiry",
    "DataDate": "date",
    "Strike": "strike",
    "Last": "last",
    "Bid": "bid",
    "Ask": "ask",
    "Volume": "volume",
    "OpenInterest": "open_interest",
    "IV": "vendor_iv",
}
_CHAIN_COLUMNS = ["date", "expiry", "underlying", "spot", "type", "strike",
                  "bid", "ask", "last", "volume", "open_interest", "vendor_iv"]
_TERM_COLUMNS = ["T", "discount", "forward", "atm_strike", "atm_iv"]


@contextmanager
def _open_text(source: Path) -> Iterator[io.TextIOBase]:
    """Text stream of the options file, from the vendor zip or the extracted CSV."""
    if not zipfile.is_zipfile(source):
        with open(source, encoding="utf-8", newline="") as fh:
            yield fh
        return
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if Path(n).name == SAMPLE_MEMBER]
        if not names:
            raise FileNotFoundError(f"{SAMPLE_MEMBER} not found in {source.name}")
        with io.TextIOWrapper(archive.open(names[0]), encoding="utf-8", newline="") as fh:
            yield fh


def read_vendor_chain(
    source: str | Path,
    underlying: str = UNDERLYING,
    valuation_date: pd.Timestamp = VALUATION_DATE,
    expiries: tuple[pd.Timestamp, ...] = EXPIRIES,
) -> tuple[pd.DataFrame, str]:
    """Select one underlying, date and set of expiries from the vendor's options file.

    ``source`` is the vendor zip or the extracted ``options_20190815.csv``.
    Returns the chain (one row per option, columns as in ``_CHAIN_COLUMNS``)
    and the SHA-256 of the selected raw lines.
    """
    wanted = {e.strftime("%m/%d/%Y") for e in expiries}
    day = valuation_date.strftime("%m/%d/%Y")
    digest = hashlib.sha256()
    records = []
    with _open_text(Path(source)) as fh:
        header = [h.strip() for h in next(csv.reader([fh.readline()]))]
        missing = set(_VENDOR_COLUMNS) - set(header)
        if missing:
            raise ValueError(f"not a Level-2 options file; missing columns {sorted(missing)}")
        index = {name: header.index(name) for name in _VENDOR_COLUMNS}
        for line in fh:
            if not line.startswith(underlying + ","):
                continue
            row = next(csv.reader([line]))
            if (row[index["UnderlyingSymbol"]] != underlying
                    or row[index["Expiration"]] not in wanted
                    or not row[index["DataDate"]].strip().startswith(day)):
                continue
            digest.update((line.rstrip("\r\n") + "\n").encode("utf-8"))
            records.append({new: row[index[old]] for old, new in _VENDOR_COLUMNS.items()})
    if not records:
        raise ValueError(f"no {underlying} rows for {day} and the requested expiries")

    chain = pd.DataFrame.from_records(records)
    chain["date"] = chain["date"].str.strip().str.slice(0, 10)
    for column in ["spot", "strike", "bid", "ask", "last", "vendor_iv"]:
        chain[column] = pd.to_numeric(chain[column])
    for column in ["volume", "open_interest"]:
        chain[column] = pd.to_numeric(chain[column]).astype("int64")
    chain["expiry"] = pd.to_datetime(chain["expiry"], format="%m/%d/%Y")
    return chain[_CHAIN_COLUMNS], digest.hexdigest()


def build_otm_quotes(chain: pd.DataFrame, valuation_date: pd.Timestamp = VALUATION_DATE) -> pd.DataFrame:
    """Forwards from put-call parity, then cleaned OTM quotes with their implied volatilities.

    One row per kept quote, sorted by expiry and strike. Expiry-level columns
    (``T, discount, forward, atm_strike, atm_iv, spot``) are repeated on every row.
    """
    chain = chain.copy()
    chain["mid"] = (chain.bid + chain.ask) / 2
    spot = float(chain.spot.iloc[0])
    term = infer_forward_discount(chain, valuation_date, spot)
    frames = []
    for expiry, row in term.iterrows():
        otm = select_otm_quotes(chain[chain.expiry == expiry], row.forward, row.discount, row["T"])
        otm.insert(0, "expiry", expiry)
        otm.insert(1, "T", row["T"])
        otm.insert(2, "discount", row.discount)
        otm.insert(3, "forward", row.forward)
        otm.insert(4, "atm_strike", row.atm_strike)
        otm.insert(5, "atm_iv", row.atm_iv)
        otm.insert(6, "spot", spot)
        frames.append(otm)
    quotes = pd.concat(frames, ignore_index=True)
    quotes.insert(0, "date", valuation_date.strftime("%Y-%m-%d"))
    quotes["expiry"] = quotes["expiry"].dt.strftime("%Y-%m-%d")
    return quotes


def load_otm_quotes(path: str | Path) -> tuple[pd.DataFrame, list[pd.DataFrame], float]:
    """Read the prepared quote file: ``(term structure, smiles per expiry, spot)``.

    pandas >= 3 parses floats exactly, so the values are bit-identical to the
    ones written by :func:`build_otm_quotes`.
    """
    quotes = pd.read_csv(path)
    quotes["expiry"] = pd.to_datetime(quotes["expiry"])
    term = quotes.groupby("expiry", sort=True)[_TERM_COLUMNS].first()
    smiles = [group.reset_index(drop=True) for _, group in quotes.groupby("expiry", sort=True)]
    return term, smiles, float(quotes.spot.iloc[0])
