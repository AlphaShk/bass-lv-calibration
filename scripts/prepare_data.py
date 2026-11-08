"""Build the cleaned SPX OTM quote file used by the notebooks.

The input is HistoricalOptionData's free end-of-day sample, which is not
redistributed with this repository:

1. On https://historicaloptiondata.com/sample-files/ download the Level 2
   end-of-day daily sample, ``Sample_L2_20190815.zip`` (about 28 MB).
2. Run ``python scripts/prepare_data.py path/to/Sample_L2_20190815.zip``
   (the extracted ``options_20190815.csv`` works too).

The script keeps SPX options quoted at the close of 15 Aug 2019 (spot 2847.60)
for the expiries 20 Sep 2019, 20 Dec 2019 and 19 Jun 2020, infers forwards and
discount factors from put-call parity, applies the OTM quote filters of
``bass_lv.market.select_otm_quotes`` and writes ``data/spx_otm_quotes_2019-08-15.csv``
(246, 65 and 67 quotes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bass_lv.data import OTM_FILE, SAMPLE_PAGE, SUBSET_SHA256, build_otm_quotes, read_vendor_chain

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / OTM_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path,
                        help="Sample_L2_20190815.zip or the extracted options_20190815.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output CSV (default: data/{OTM_FILE})")
    args = parser.parse_args(argv)
    if not args.source.exists():
        parser.error(f"{args.source} not found; download it from {SAMPLE_PAGE}")

    chain, digest = read_vendor_chain(args.source)
    print(f"selected {len(chain):,} SPX calls and puts "
          f"({chain.date.iloc[0]}, spot {chain.spot.iloc[0]:.2f})")
    if digest == SUBSET_SHA256:
        print("input check: identical to the subset behind the published results")
    else:
        print("input check: WARNING, the selected rows differ from the subset behind "
              "the published results; numbers may not match the README", file=sys.stderr)

    quotes = build_otm_quotes(chain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(args.out, index=False)

    summary = (quotes.groupby("expiry")
               .agg(quotes=("strike", "size"), forward=("forward", "first"),
                    discount=("discount", "first"), atm_iv=("atm_iv", "first")))
    print(summary.round({"forward": 2, "discount": 5, "atm_iv": 4}).to_string())
    print(f"wrote {len(quotes)} OTM quotes to {args.out.name} in {args.out.parent.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
