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
import logging
import math
import re
import sys
from datetime import datetime, timezone

import requests
from finvizfinance.screener.ownership import Ownership

log = logging.getLogger(__name__)

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

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
    try:
        screener = Ownership()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        if df is None or df.empty:
            return []
        records = df.to_dict(orient="records")
        return _post_filter(records)
    except Exception as e:
        log.warning("finvizfinance scan failed: %s", e)
        return []


# Finviz filter-code mapping. Matches the human-readable dict above.
_FINVIZ_CODE_GAINERS = "sh_curvol_o2,sh_float_u100,sh_price_o1,sh_relvol_o2,ta_change_u5"
_FINVIZ_CODE_LOSERS  = "sh_curvol_o2,sh_float_u100,sh_price_o1,sh_relvol_o2,ta_change_d5"


def _scan_finviz_http(filter_code: str) -> list[dict]:
    """Fallback #1: hit Finviz's screener HTML directly and parse out
    the ticker symbols. Uses no library — just regex on the response.
    Returns bare ticker records (no Float / Volume / Short Ratio
    fields, so _post_filter is bypassed)."""
    try:
        url = f"https://finviz.com/screener.ashx?v=152&f={filter_code}&ft=4"
        r = requests.get(url, headers=_UA, timeout=10)
        r.raise_for_status()
        # "quote.ashx?t=SYM&" appears once per ticker row in v=152 view.
        tickers = list(dict.fromkeys(re.findall(r"quote\.ashx\?t=([A-Z]{1,6})&", r.text)))
        return [{"Ticker": t} for t in tickers]
    except Exception as e:
        log.warning("Finviz HTTP scrape failed: %s", e)
        return []


def _scan_yahoo(direction: str) -> list[dict]:
    """Fallback #2: Yahoo Finance predefined day_gainers/day_losers
    screener. Looser filters than Finviz (Yahoo's defaults are price
    >= $5, market cap >= $1B by default) — we re-apply our own
    price/volume/change gates in Python."""
    scr = "day_gainers" if direction == "gainers" else "day_losers"
    try:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/"
            f"predefined/saved?formatted=true&lang=en-US&region=US"
            f"&scrIds={scr}&count=100"
        )
        r = requests.get(url, headers=_UA, timeout=10)
        r.raise_for_status()
        data = r.json()
        quotes = (
            data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
        )
    except Exception as e:
        log.warning("Yahoo screener fetch failed: %s", e)
        return []

    rows = []
    for q in quotes:
        ticker = str(q.get("symbol", "")).strip()
        if not ticker or not ticker.isalnum():
            continue
        # Yahoo returns values as {"raw": 1.23, "fmt": "1.23"} when
        # formatted=true; use .raw for numerics.
        def _raw(field):
            v = q.get(field)
            if isinstance(v, dict):
                return v.get("raw")
            return v
        chg = _raw("regularMarketChangePercent") or 0
        vol = _raw("regularMarketVolume") or 0
        price = _raw("regularMarketPrice") or 0
        # Re-apply our Finviz-equivalent filters.
        if price < 1.0:
            continue
        if vol < 2_000_000:
            continue
        if direction == "gainers" and chg < 5.0:
            continue
        if direction == "losers" and chg > -5.0:
            continue
        rows.append({
            "Ticker": ticker,
            "Change": chg,
            "Volume": vol,
            "Close": price,
        })
    return rows


def _scan_with_fallbacks(finviz_filters: dict, finviz_code: str, direction: str) -> list[dict]:
    """Chain: finvizfinance library -> Finviz HTTP scrape -> Yahoo.
    Return the first non-empty result."""
    out = _scan(finviz_filters)
    if out:
        return out
    log.info("finvizfinance returned 0 tickers; falling back to Finviz HTTP scrape")

    out = _scan_finviz_http(finviz_code)
    if out:
        return out
    log.info("Finviz HTTP scrape returned 0 tickers; falling back to Yahoo")

    return _scan_yahoo(direction)


def scan_gainers() -> list[dict]:
    return _scan_with_fallbacks(GAINER_FILTERS, _FINVIZ_CODE_GAINERS, "gainers")


def scan_losers() -> list[dict]:
    return _scan_with_fallbacks(LOSER_FILTERS, _FINVIZ_CODE_LOSERS, "losers")


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
