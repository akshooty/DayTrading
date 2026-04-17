"""Finviz premarket scanners for breakout-long (gainers) and
breakout-short (losers) universes.

Universe filters (Finviz-side):
  - Change >= 5% (gainers) / <= -5% (losers)
  - Current Volume >= 2M shares
  - Price >= $1
  - Relative Volume >= 2x (today's volume vs 50-day avg)
  - Shares Float < 100M (low-float movers preferred)

Universe filters (post-scan):
  - Short Ratio > 4 (days-to-cover; squeeze fuel)
  - Float/Volume < 20 (float rotating heavily today)
"""

import json
import math
import sys
from datetime import datetime, timezone

from finvizfinance.screener.ownership import Ownership

GAINER_FILTERS = {
    "Change": "Up 5%",
    "Current Volume": "Over 2M",
    "Price": "Over $1",
    "Relative Volume": "Over 2",
    "Float": "Under 100M",
}

LOSER_FILTERS = {
    "Change": "Down 5%",
    "Current Volume": "Over 2M",
    "Price": "Over $1",
    "Relative Volume": "Over 2",
    "Float": "Under 100M",
}

MAX_FLOAT_TO_VOLUME = 20.0


def _parse_number(x) -> float | None:
    """Parse Finviz values like '25.30M', '1.2B', '3.45%', '-', or bare numbers."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if not (isinstance(x, float) and math.isnan(x)) else None
    s = str(x).strip()
    if s in ("", "-", "N/A"):
        return None
    suffix = s[-1]
    try:
        if suffix == "K":
            return float(s[:-1]) * 1_000
        if suffix == "M":
            return float(s[:-1]) * 1_000_000
        if suffix == "B":
            return float(s[:-1]) * 1_000_000_000
        if suffix == "%":
            return float(s[:-1]) / 100
        return float(s)
    except ValueError:
        return None


def _post_filter(records: list[dict]) -> list[dict]:
    """Apply the float/volume gate, then sort by short ratio (desc).

    Short ratio is treated as a RANKING signal, not a hard filter — the
    trader subscribes in this order so when multiple ARM signals compete
    for the concurrent-position cap, higher-squeeze-fuel names win ties.
    """
    kept = []
    for r in records:
        float_shs = _parse_number(r.get("Float"))
        volume = _parse_number(r.get("Volume"))
        if float_shs is None or volume is None or volume == 0:
            continue
        if float_shs / volume >= MAX_FLOAT_TO_VOLUME:
            continue
        kept.append(r)

    def sort_key(r):
        sr = _parse_number(r.get("Short Ratio"))
        # Missing SR → sort last. Otherwise higher SR first.
        return (sr is None, -(sr or 0.0))

    kept.sort(key=sort_key)
    return kept


def _scan(filters: dict) -> list[dict]:
    screener = Ownership()
    screener.set_filter(filters_dict=filters)
    df = screener.screener_view()
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return _post_filter(records)


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
            "post_filters": {"max_float_to_volume": MAX_FLOAT_TO_VOLUME},
            "ranked_by": "short_ratio_desc",
            "count": len(gainers),
            "tickers": gainers,
        },
        "losers": {
            "filters": LOSER_FILTERS,
            "post_filters": {"max_float_to_volume": MAX_FLOAT_TO_VOLUME},
            "ranked_by": "short_ratio_desc",
            "count": len(losers),
            "tickers": losers,
        },
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
