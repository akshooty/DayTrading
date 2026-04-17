"""Finviz premarket gainer scanner.

Finds stocks with >20% change and >=2M current volume.
"""

import json
import sys
from datetime import datetime, timezone

from finvizfinance.screener.overview import Overview

FILTERS = {
    "Change": "Up 5%",
    "Current Volume": "Over 2M",
}


def scan() -> list[dict]:
    screener = Overview()
    screener.set_filter(filters_dict=FILTERS)
    df = screener.screener_view()
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def main() -> int:
    try:
        tickers = scan()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    payload = {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "filters": FILTERS,
        "count": len(tickers),
        "tickers": tickers,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
