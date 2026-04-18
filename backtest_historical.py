"""Back-test the strategy against a historical trading day.

Usage:
    DATE=2026-04-16 python backtest_historical.py

Finds that day's gainers and losers from Alpaca's historical data
(>=5% change, >=2M volume, price >= $1), then replays the breakout
strategies against that day's 1-min bars.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from dotenv import load_dotenv

EASTERN = ZoneInfo("America/New_York")
MIN_CHANGE_PCT = 5.0
MIN_VOLUME = 2_000_000
MIN_PRICE = float(os.environ.get("MIN_PRICE", 1.0))
MAX_PRICE = float(os.environ.get("MAX_PRICE", 1e9))
TOP_N = 50  # cap universe to top-N movers per side to keep bar fetch reasonable


def prior_trading_day(d: date) -> date:
    p = d - timedelta(days=1)
    while p.weekday() >= 5:
        p -= timedelta(days=1)
    return p


def fetch_daily_bars(start_date: date, end_date: date) -> dict[str, list]:
    """Fetch daily bars for all tradable US equities over a date range.
    Used once upfront to avoid re-fetching per-day in multi-day backtests.
    """
    load_dotenv()
    dclient = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"].strip(), os.environ["ALPACA_SECRET_KEY"].strip()
    )
    tclient = TradingClient(
        os.environ["ALPACA_API_KEY"].strip(), os.environ["ALPACA_SECRET_KEY"].strip(), paper=True
    )
    assets = tclient.get_all_assets(GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY,
    ))
    symbols = [
        a.symbol for a in assets
        if a.tradable and a.symbol and a.symbol.isalnum()
    ]
    print(f"universe: {len(symbols)} tradable US equities", flush=True)

    start = datetime.combine(start_date, time(0, 0), tzinfo=EASTERN)
    end = datetime.combine(end_date + timedelta(days=1), time(0, 0), tzinfo=EASTERN)
    feed = DataFeed[os.environ.get("FEED", "IEX").upper()]

    all_bars: dict[str, list] = {}
    chunk = 500
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            resp = dclient.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start, end=end,
                feed=feed,
            ))
            for sym, bars in resp.data.items():
                all_bars[sym] = bars
        except Exception as e:
            print(f"  chunk {i}-{i+chunk} failed: {e}", file=sys.stderr, flush=True)
        print(f"  fetched {i+len(batch)}/{len(symbols)} symbols", flush=True)
    print(f"got daily bars for {len(all_bars)} symbols", flush=True)
    return all_bars


def movers_from_cache(target_date: date, all_bars: dict[str, list]) -> tuple[list[dict], list[dict]]:
    prev = prior_trading_day(target_date)
    gainers: list[dict] = []
    losers: list[dict] = []
    for sym, bars in all_bars.items():
        if len(bars) < 2:
            continue
        by_date = {b.timestamp.astimezone(EASTERN).date(): b for b in bars}
        if target_date not in by_date or prev not in by_date:
            continue
        target_bar = by_date[target_date]
        prev_bar = by_date[prev]
        if target_bar.close < MIN_PRICE or target_bar.close > MAX_PRICE or target_bar.volume < MIN_VOLUME:
            continue
        change = (target_bar.close - prev_bar.close) / prev_bar.close * 100
        row = {
            "Ticker": sym, "Change": change, "Volume": target_bar.volume,
            "Close": target_bar.close,
        }
        no_gap_down_longs = os.environ.get("NO_GAP_DOWN_LONGS", "").lower() in ("1", "true", "yes")
        if change >= MIN_CHANGE_PCT:
            if no_gap_down_longs and target_bar.open < prev_bar.close:
                continue
            gainers.append(row)
        elif change <= -MIN_CHANGE_PCT:
            losers.append(row)
    gainers.sort(key=lambda r: -r["Change"])
    losers.sort(key=lambda r: r["Change"])
    return gainers[:TOP_N], losers[:TOP_N]


def find_movers(target_date: date) -> tuple[list[dict], list[dict]]:
    """Single-date convenience wrapper — fetches daily bars for just
    target_date and its prior trading day, then extracts movers."""
    prev = prior_trading_day(target_date)
    all_bars = fetch_daily_bars(prev, target_date)
    return movers_from_cache(target_date, all_bars)


def main() -> int:
    target = os.environ.get("DATE")
    if target:
        target_date = date.fromisoformat(target)
    else:
        target_date = prior_trading_day(date.today())
    print(f"target date: {target_date} ({target_date.strftime('%A')})")
    print()

    gainers, losers = find_movers(target_date)
    print(f"gainers (>= +{MIN_CHANGE_PCT}%): {len(gainers)}")
    for g in gainers[:15]:
        print(f"  {g['Ticker']:<6} {g['Change']:+6.2f}%  vol {g['Volume']:>15,.0f}  ${g['Close']:.2f}")
    print(f"losers  (<= -{MIN_CHANGE_PCT}%): {len(losers)}")
    for l in losers[:15]:
        print(f"  {l['Ticker']:<6} {l['Change']:+6.2f}%  vol {l['Volume']:>15,.0f}  ${l['Close']:.2f}")
    print()

    long_tickers = ",".join(g["Ticker"] for g in gainers)
    short_tickers = ",".join(l["Ticker"] for l in losers)

    # Re-exec backtest.py with the override + a BACKTEST_DATE env so it
    # scopes bar fetch to the historical session.
    env = os.environ.copy()
    env["LONG_WATCHLIST"] = long_tickers
    env["SHORT_WATCHLIST"] = short_tickers
    env["BACKTEST_DATE"] = target_date.isoformat()
    result = subprocess.run(
        [sys.executable, "backtest.py"],
        env=env, check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
