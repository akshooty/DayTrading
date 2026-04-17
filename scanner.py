"""Finviz premarket scanners for breakout-long (gainers) and
breakout-short (losers) universes.
"""

import json
import sys
from datetime import datetime, timezone

from finvizfinance.screener.overview import Overview

GAINER_FILTERS = {
    "Change": "Up 5%",
    "Current Volume": "Over 2M",
    "Price": "Over $5",
}

LOSER_FILTERS = {
    "Change": "Down 5%",
    "Current Volume": "Over 2M",
    "Price": "Over $5",
}

EXCLUDED_INDUSTRIES = {"Exchange Traded Fund"}


def _scan(filters: dict) -> list[dict]:
    screener = Overview()
    screener.set_filter(filters_dict=filters)
    df = screener.screener_view()
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return [r for r in records if r.get("Industry") not in EXCLUDED_INDUSTRIES]


def scan_gainers() -> list[dict]:
    return _scan(GAINER_FILTERS)


def scan_losers() -> list[dict]:
    return _scan(LOSER_FILTERS)


# Legacy alias — used by the remote scheduled agent's prompt.
def scan() -> list[dict]:
    return scan_gainers()


def main() -> int:
    try:
        gainers = scan_gainers()
        losers = scan_losers()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    payload = {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "gainers": {
            "filters": GAINER_FILTERS,
            "count": len(gainers),
            "tickers": gainers,
        },
        "losers": {
            "filters": LOSER_FILTERS,
            "count": len(losers),
            "tickers": losers,
        },
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
