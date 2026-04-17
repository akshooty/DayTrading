"""Replay the breakout-long strategy against historical 1-min bars.

Defaults to today's Finviz scan universe from market open through now.
Override watchlist with WATCHLIST env var (comma-separated tickers).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from scanner import scan
from strategy import (
    ArmSignal,
    Bar as IntBar,
    BreakoutLongState,
    ResetSignal,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
RISK_PCT = 0.005
MIN_STOP_DIST = 0.10
MAX_CONCURRENT = 5
LIMIT_OFFSET = 0.05


@dataclass
class Trade:
    ticker: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    opened_at: datetime
    closed_at: datetime
    status: str  # 'trail_stop' | 'eod'


def load_watchlist() -> list[str]:
    override = os.environ.get("WATCHLIST")
    if override:
        return [s.strip().upper() for s in override.split(",") if s.strip()]
    return [t["Ticker"] for t in scan()]


def run_backtest():
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )

    equity = float(os.environ.get("BACKTEST_EQUITY", "953000"))

    tickers = load_watchlist()
    print(f"backtesting {len(tickers)} tickers")

    now_et = datetime.now(EASTERN)
    start_et = datetime.combine(now_et.date(), time(9, 30), tzinfo=EASTERN)
    end_et = now_et

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Minute,
        start=start_et,
        end=end_et,
        feed=DataFeed.IEX,
    )
    resp = client.get_stock_bars(req)

    events = []
    for ticker, bars in resp.data.items():
        for b in bars:
            events.append((b.timestamp, ticker, b))
    events.sort(key=lambda x: x[0])
    print(f"loaded {len(events)} bar events from {start_et.time()} ET to {end_et.time()} ET")
    print()

    states = {t: BreakoutLongState(t) for t in tickers}
    positions: dict[str, dict] = {}
    pending: dict[str, ArmSignal] = {}
    trades: list[Trade] = []
    skipped_at_cap = 0

    for ts, ticker, b in events:
        st = states.get(ticker)
        if st is None:
            continue

        # Fill pending entry if this bar crosses the stop price
        if ticker in pending:
            sig = pending[ticker]
            if b.high >= sig.entry:
                fill = max(sig.entry, b.open)
                qty = position_size(equity, RISK_PCT, sig.entry, sig.stop, MIN_STOP_DIST)
                if qty > 0:
                    trail = max(fill - sig.stop, MIN_STOP_DIST)
                    positions[ticker] = dict(
                        entry=fill,
                        qty=qty,
                        high=fill,
                        trail=trail,
                        stop=fill - trail,
                        opened_at=ts,
                    )
                    st.on_entry_filled()
                del pending[ticker]

        # Manage open position — update trail, check exit
        if ticker in positions:
            pos = positions[ticker]
            if b.high > pos["high"]:
                pos["high"] = b.high
                pos["stop"] = pos["high"] - pos["trail"]
            if b.low <= pos["stop"]:
                exit_px = pos["stop"]
                pnl = (exit_px - pos["entry"]) * pos["qty"]
                trades.append(Trade(
                    ticker=ticker, entry_price=pos["entry"], exit_price=exit_px,
                    qty=pos["qty"], pnl=pnl,
                    opened_at=pos["opened_at"], closed_at=ts, status="trail_stop",
                ))
                del positions[ticker]
                st.on_exit_filled()

        # Drive state machine
        ib = IntBar(
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=b.volume, timestamp=ts,
        )
        sig = st.on_bar(ib)
        if isinstance(sig, ArmSignal):
            if len(positions) + len(pending) >= MAX_CONCURRENT:
                skipped_at_cap += 1
                st.reset_to_watching()
            else:
                pending[ticker] = sig
        elif isinstance(sig, ResetSignal):
            pending.pop(ticker, None)

    # Mark-to-market any still-open positions at the final bar close
    last_bars: dict[str, float] = {}
    for _, ticker, b in events:
        last_bars[ticker] = b.close
    for ticker, pos in positions.items():
        last = last_bars.get(ticker, pos["entry"])
        pnl = (last - pos["entry"]) * pos["qty"]
        trades.append(Trade(
            ticker=ticker, entry_price=pos["entry"], exit_price=last,
            qty=pos["qty"], pnl=pnl,
            opened_at=pos["opened_at"], closed_at=end_et, status="open_eod",
        ))

    # Report
    print(f"{'Ticker':<7} {'Entry':>7} {'Exit':>7} {'Qty':>6} {'P&L':>10}  {'Status':<10}  Duration")
    print("-" * 75)
    for t in sorted(trades, key=lambda x: x.opened_at):
        dur = t.closed_at - t.opened_at
        print(
            f"{t.ticker:<7} {t.entry_price:>7.2f} {t.exit_price:>7.2f} "
            f"{t.qty:>6} ${t.pnl:>9.2f}  {t.status:<10}  {dur}"
        )

    closed_pnl = sum(t.pnl for t in trades if t.status == "trail_stop")
    open_pnl = sum(t.pnl for t in trades if t.status == "open_eod")
    print()
    print(f"Closed trades:   {sum(1 for t in trades if t.status == 'trail_stop')}  (P&L ${closed_pnl:+,.2f})")
    print(f"Still open:      {sum(1 for t in trades if t.status == 'open_eod')}  (mark-to-market ${open_pnl:+,.2f})")
    print(f"ARM skipped (at cap={MAX_CONCURRENT}): {skipped_at_cap}")
    print(f"Total P&L:       ${closed_pnl + open_pnl:+,.2f}  on ${equity:,.0f} equity ({(closed_pnl + open_pnl)/equity*100:+.3f}%)")


if __name__ == "__main__":
    run_backtest()
