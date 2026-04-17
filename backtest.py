"""Replay breakout-long + breakout-short strategies against historical
1-min bars. Defaults: today's Finviz gainers + losers from market open
through now. WATCHLIST env override applies LONG-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from scanner import scan_gainers, scan_losers
from strategy import (
    ArmSignal,
    Bar as IntBar,
    BreakoutLongState,
    BreakoutShortState,
    Direction,
    ResetSignal,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
RISK_PCT = 0.005
MIN_STOP_DIST = 0.10
MAX_CONCURRENT = 8
MAX_DEPLOYMENT = 5000.0


@dataclass
class Trade:
    ticker: str
    direction: Direction
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    opened_at: datetime
    closed_at: datetime
    status: str  # 'trail_stop' | 'open_eod'


def _valid(sym: str) -> bool:
    # Alpaca only accepts plain alphanumeric tickers; drop things like PBR-A.
    return bool(sym) and sym.isalnum()


def load_watchlist() -> dict[str, Direction]:
    override = os.environ.get("WATCHLIST")
    if override:
        return {s.strip().upper(): Direction.LONG for s in override.split(",") if _valid(s.strip().upper())}
    out: dict[str, Direction] = {}
    for t in scan_gainers():
        sym = t.get("Ticker") or ""
        if _valid(sym):
            out[sym] = Direction.LONG
    for t in scan_losers():
        sym = t.get("Ticker") or ""
        if _valid(sym) and sym not in out:
            out[sym] = Direction.SHORT
    return out


def run_backtest():
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    equity = float(os.environ.get("BACKTEST_EQUITY", "953000"))

    watchlist = load_watchlist()
    longs = [s for s, d in watchlist.items() if d is Direction.LONG]
    shorts = [s for s, d in watchlist.items() if d is Direction.SHORT]
    print(f"backtesting: {len(longs)} LONG + {len(shorts)} SHORT = {len(watchlist)} tickers")

    now_et = datetime.now(EASTERN)
    start_et = datetime.combine(now_et.date(), time(9, 30), tzinfo=EASTERN)
    end_et = now_et

    req = StockBarsRequest(
        symbol_or_symbols=list(watchlist.keys()),
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
    print(f"loaded {len(events)} bar events")
    print()

    states: dict = {}
    for sym, d in watchlist.items():
        states[sym] = BreakoutLongState(sym) if d is Direction.LONG else BreakoutShortState(sym)

    positions: dict[str, dict] = {}
    pending: dict[str, ArmSignal] = {}
    trades: list[Trade] = []
    skipped_at_cap = 0

    for ts, ticker, b in events:
        st = states.get(ticker)
        if st is None:
            continue

        # Fill pending entry if this bar touches the trigger
        if ticker in pending:
            sig = pending[ticker]
            filled = False
            if sig.direction is Direction.LONG and b.high >= sig.entry:
                fill = max(sig.entry, b.open)
                filled = True
            elif sig.direction is Direction.SHORT and b.low <= sig.entry:
                fill = min(sig.entry, b.open)
                filled = True

            if filled:
                qty = position_size(
                    equity, RISK_PCT, sig.entry, sig.stop,
                    MIN_STOP_DIST, MAX_DEPLOYMENT,
                )
                if qty > 0:
                    if sig.direction is Direction.LONG:
                        trail = max(fill - sig.stop, MIN_STOP_DIST)
                        stop = fill - trail
                        extreme = fill  # tracks highest since fill
                    else:
                        trail = max(sig.stop - fill, MIN_STOP_DIST)
                        stop = fill + trail
                        extreme = fill  # tracks lowest since fill
                    positions[ticker] = dict(
                        direction=sig.direction,
                        entry=fill, qty=qty, extreme=extreme,
                        trail=trail, stop=stop, opened_at=ts,
                    )
                    st.on_entry_filled()
                del pending[ticker]

        # Manage open position — update trail and check exit
        if ticker in positions:
            pos = positions[ticker]
            if pos["direction"] is Direction.LONG:
                if b.high > pos["extreme"]:
                    pos["extreme"] = b.high
                    pos["stop"] = pos["extreme"] - pos["trail"]
                exited = b.low <= pos["stop"]
                exit_px = pos["stop"] if exited else None
            else:
                if b.low < pos["extreme"]:
                    pos["extreme"] = b.low
                    pos["stop"] = pos["extreme"] + pos["trail"]
                exited = b.high >= pos["stop"]
                exit_px = pos["stop"] if exited else None

            if exited:
                pnl = (
                    (exit_px - pos["entry"]) * pos["qty"]
                    if pos["direction"] is Direction.LONG
                    else (pos["entry"] - exit_px) * pos["qty"]
                )
                trades.append(Trade(
                    ticker=ticker, direction=pos["direction"],
                    entry_price=pos["entry"], exit_price=exit_px,
                    qty=pos["qty"], pnl=pnl,
                    opened_at=pos["opened_at"], closed_at=ts,
                    status="trail_stop",
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

    # Mark-to-market still-open positions at last close per ticker
    last_close: dict[str, float] = {}
    for _, ticker, b in events:
        last_close[ticker] = b.close
    for ticker, pos in positions.items():
        last = last_close.get(ticker, pos["entry"])
        if pos["direction"] is Direction.LONG:
            pnl = (last - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - last) * pos["qty"]
        trades.append(Trade(
            ticker=ticker, direction=pos["direction"],
            entry_price=pos["entry"], exit_price=last,
            qty=pos["qty"], pnl=pnl,
            opened_at=pos["opened_at"], closed_at=end_et,
            status="open_eod",
        ))

    print(f"{'Ticker':<7} {'Dir':<5} {'Entry':>7} {'Exit':>7} {'Qty':>6} {'P&L':>10}  {'Status':<10}  Duration")
    print("-" * 85)
    for t in sorted(trades, key=lambda x: x.opened_at):
        dur = t.closed_at - t.opened_at
        print(
            f"{t.ticker:<7} {t.direction.value:<5} {t.entry_price:>7.2f} {t.exit_price:>7.2f} "
            f"{t.qty:>6} ${t.pnl:>9.2f}  {t.status:<10}  {dur}"
        )

    closed_pnl = sum(t.pnl for t in trades if t.status == "trail_stop")
    open_pnl = sum(t.pnl for t in trades if t.status == "open_eod")
    by_dir = {}
    for t in trades:
        by_dir.setdefault(t.direction, 0.0)
        by_dir[t.direction] += t.pnl
    print()
    print(f"Closed:    {sum(1 for t in trades if t.status == 'trail_stop')}  P&L ${closed_pnl:+,.2f}")
    print(f"Open MTM:  {sum(1 for t in trades if t.status == 'open_eod')}  P&L ${open_pnl:+,.2f}")
    print(f"Skipped at cap={MAX_CONCURRENT}: {skipped_at_cap}")
    for d, pnl in by_dir.items():
        print(f"  {d.value:<5} P&L ${pnl:+,.2f}")
    total = closed_pnl + open_pnl
    print(f"Total:     ${total:+,.2f}  ({total/equity*100:+.3f}% of ${equity:,.0f})")


if __name__ == "__main__":
    run_backtest()
