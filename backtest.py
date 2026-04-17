"""Replay breakout-long + breakout-short against intraday 1-min bars.

Exit logic:
  Phase 1 (0 -> +1R):   static stop at initial_stop (pullback_low /
                        bounce_high). Wide room for the trade to work.
  Phase 2 (+1R hit):    sell 50% at partial take-profit. Move stop for
                        the remaining 50% to breakeven. Trailing from
                        there with trail_dist = initial_risk.

Universe: scan_gainers() + scan_losers() (both already include price
>=$5 and exclude ETFs). WATCHLIST env override = LONG-only list.

Guardrails active during backtest:
  - Time-of-day filter: no NEW entries 11:30 AM - 2:00 PM ET.
  - Daily loss circuit breaker: if realized P&L <= -1% of equity, stop
    taking new entries for the rest of the day.
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
    min_stop_dist,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
MAX_CONCURRENT = 8
MAX_DEPLOYMENT = 5000.0
MAX_RISK_PER_TRADE = 200.0
CHOP_START = time(11, 30)
CHOP_END = time(14, 0)
DAILY_LOSS_LIMIT_PCT = 0.01


@dataclass
class Trade:
    ticker: str
    direction: Direction
    entry_price: float
    original_qty: int
    tp_price: float | None
    tp_qty: int
    final_exit_price: float
    final_exit_qty: int
    pnl: float
    opened_at: datetime
    closed_at: datetime
    status: str  # 'closed_with_tp' | 'closed_no_tp' | 'open_eod'


def _valid(sym: str) -> bool:
    return bool(sym) and sym.isalnum()


def load_watchlist() -> dict[str, Direction]:
    longs = os.environ.get("LONG_WATCHLIST", "")
    shorts = os.environ.get("SHORT_WATCHLIST", "")
    if longs or shorts:
        out: dict[str, Direction] = {}
        for s in longs.split(","):
            sym = s.strip().upper()
            if _valid(sym):
                out[sym] = Direction.LONG
        for s in shorts.split(","):
            sym = s.strip().upper()
            if _valid(sym) and sym not in out:
                out[sym] = Direction.SHORT
        return out

    override = os.environ.get("WATCHLIST")
    if override:
        return {s.strip().upper(): Direction.LONG for s in override.split(",") if _valid(s.strip().upper())}

    out = {}
    for t in scan_gainers():
        sym = t.get("Ticker") or ""
        if _valid(sym):
            out[sym] = Direction.LONG
    for t in scan_losers():
        sym = t.get("Ticker") or ""
        if _valid(sym) and sym not in out:
            out[sym] = Direction.SHORT
    return out


def _in_chop(ts_utc: datetime) -> bool:
    t = ts_utc.astimezone(EASTERN).time()
    return CHOP_START <= t < CHOP_END


def run_backtest():
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    equity = float(os.environ.get("BACKTEST_EQUITY", "953000"))
    daily_loss_cap = -equity * DAILY_LOSS_LIMIT_PCT

    watchlist = load_watchlist()
    longs = [s for s, d in watchlist.items() if d is Direction.LONG]
    shorts = [s for s, d in watchlist.items() if d is Direction.SHORT]
    print(f"backtesting: {len(longs)} LONG + {len(shorts)} SHORT = {len(watchlist)} tickers")

    target = os.environ.get("BACKTEST_DATE")
    if target:
        from datetime import date as _date
        d = _date.fromisoformat(target)
        start_et = datetime.combine(d, time(9, 30), tzinfo=EASTERN)
        end_et = datetime.combine(d, time(16, 0), tzinfo=EASTERN)
    else:
        now_et = datetime.now(EASTERN)
        start_et = datetime.combine(now_et.date(), time(9, 30), tzinfo=EASTERN)
        end_et = now_et

    feed = DataFeed[os.environ.get("FEED", "IEX").upper()]
    req = StockBarsRequest(
        symbol_or_symbols=list(watchlist.keys()),
        timeframe=TimeFrame.Minute,
        start=start_et,
        end=end_et,
        feed=feed,
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
    skipped_chop = 0
    circuit_broken = False
    realized_pnl = 0.0

    for ts, ticker, b in events:
        st = states.get(ticker)
        if st is None:
            continue

        # Check pending fill
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
                    sig.entry, sig.stop,
                    max_deployment=MAX_DEPLOYMENT,
                    max_risk=MAX_RISK_PER_TRADE,
                )
                if qty >= 2:  # need at least 2 to split 50/50
                    initial_risk = max(abs(fill - sig.stop), min_stop_dist(fill))
                    if sig.direction is Direction.LONG:
                        tp_level = fill + initial_risk
                    else:
                        tp_level = fill - initial_risk
                    positions[ticker] = dict(
                        direction=sig.direction,
                        entry=fill,
                        original_qty=qty,
                        remaining_qty=qty,
                        initial_risk=initial_risk,
                        initial_stop=sig.stop,
                        tp_level=tp_level,
                        phase=1,
                        extreme=fill,
                        stop=sig.stop,
                        tp_price=None,
                        opened_at=ts,
                    )
                    st.on_entry_filled()
                del pending[ticker]

        # Manage open position
        if ticker in positions:
            pos = positions[ticker]
            d = pos["direction"]

            # Phase 1: check for TP level hit
            if pos["phase"] == 1:
                hit_tp = (
                    (d is Direction.LONG and b.high >= pos["tp_level"])
                    or (d is Direction.SHORT and b.low <= pos["tp_level"])
                )
                if hit_tp:
                    half = pos["original_qty"] // 2
                    tp_pnl = (
                        (pos["tp_level"] - pos["entry"]) * half
                        if d is Direction.LONG
                        else (pos["entry"] - pos["tp_level"]) * half
                    )
                    pos["tp_price"] = pos["tp_level"]
                    pos["remaining_qty"] = pos["original_qty"] - half
                    pos["phase"] = 2
                    pos["stop"] = pos["entry"]  # breakeven on remaining
                    pos["extreme"] = pos["tp_level"]
                    pos["_tp_pnl"] = tp_pnl
                    pos["_tp_qty"] = half
                    realized_pnl += tp_pnl

            # Phase 2: trailing stop for remaining 50%
            if pos["phase"] == 2:
                if d is Direction.LONG:
                    if b.high > pos["extreme"]:
                        pos["extreme"] = b.high
                        pos["stop"] = max(pos["stop"], pos["extreme"] - pos["initial_risk"])
                else:
                    if b.low < pos["extreme"]:
                        pos["extreme"] = b.low
                        pos["stop"] = min(pos["stop"], pos["extreme"] + pos["initial_risk"])

            # Check exit (phase 1 = initial static stop, phase 2 = trailing)
            if d is Direction.LONG:
                exited = b.low <= pos["stop"]
            else:
                exited = b.high >= pos["stop"]

            if exited:
                exit_px = pos["stop"]
                final_qty = pos["remaining_qty"]
                final_pnl = (
                    (exit_px - pos["entry"]) * final_qty
                    if d is Direction.LONG
                    else (pos["entry"] - exit_px) * final_qty
                )
                tp_pnl = pos.get("_tp_pnl", 0.0)
                tp_qty = pos.get("_tp_qty", 0)
                realized_pnl += final_pnl
                status = "closed_with_tp" if pos["tp_price"] is not None else "closed_no_tp"
                trades.append(Trade(
                    ticker=ticker, direction=d,
                    entry_price=pos["entry"],
                    original_qty=pos["original_qty"],
                    tp_price=pos["tp_price"],
                    tp_qty=tp_qty,
                    final_exit_price=exit_px,
                    final_exit_qty=final_qty,
                    pnl=tp_pnl + final_pnl,
                    opened_at=pos["opened_at"], closed_at=ts,
                    status=status,
                ))
                del positions[ticker]
                st.on_exit_filled()

        # Circuit breaker — stop taking new entries
        if realized_pnl <= daily_loss_cap and not circuit_broken:
            circuit_broken = True

        # Drive state machine
        ib = IntBar(
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=b.volume, timestamp=ts,
        )
        sig = st.on_bar(ib)
        if isinstance(sig, ArmSignal):
            if circuit_broken:
                st.reset_to_watching()
                continue
            if _in_chop(ts):
                skipped_chop += 1
                st.reset_to_watching()
                continue
            if len(positions) + len(pending) >= MAX_CONCURRENT:
                skipped_at_cap += 1
                st.reset_to_watching()
                continue
            pending[ticker] = sig
        elif isinstance(sig, ResetSignal):
            pending.pop(ticker, None)

    # Mark-to-market still-open
    last_close: dict[str, float] = {}
    for _, ticker, b in events:
        last_close[ticker] = b.close
    for ticker, pos in positions.items():
        last = last_close.get(ticker, pos["entry"])
        d = pos["direction"]
        final_pnl = (
            (last - pos["entry"]) * pos["remaining_qty"]
            if d is Direction.LONG
            else (pos["entry"] - last) * pos["remaining_qty"]
        )
        tp_pnl = pos.get("_tp_pnl", 0.0)
        tp_qty = pos.get("_tp_qty", 0)
        trades.append(Trade(
            ticker=ticker, direction=d,
            entry_price=pos["entry"],
            original_qty=pos["original_qty"],
            tp_price=pos["tp_price"],
            tp_qty=tp_qty,
            final_exit_price=last,
            final_exit_qty=pos["remaining_qty"],
            pnl=tp_pnl + final_pnl,
            opened_at=pos["opened_at"], closed_at=end_et,
            status="open_eod",
        ))

    # Report
    print(
        f"{'Ticker':<7} {'Dir':<5} {'Entry':>7} {'TP':>7} {'Exit':>7} "
        f"{'Qty':>6} {'P&L':>10}  {'Status':<15} Duration"
    )
    print("-" * 90)
    for t in sorted(trades, key=lambda x: x.opened_at):
        dur = t.closed_at - t.opened_at
        tp_str = f"{t.tp_price:.2f}" if t.tp_price is not None else "-"
        print(
            f"{t.ticker:<7} {t.direction.value:<5} {t.entry_price:>7.2f} "
            f"{tp_str:>7} {t.final_exit_price:>7.2f} {t.original_qty:>6} "
            f"${t.pnl:>9.2f}  {t.status:<15} {dur}"
        )

    closed_pnl = sum(t.pnl for t in trades if t.status != "open_eod")
    open_pnl = sum(t.pnl for t in trades if t.status == "open_eod")
    by_dir = {}
    for t in trades:
        by_dir.setdefault(t.direction, 0.0)
        by_dir[t.direction] += t.pnl
    n_closed = sum(1 for t in trades if t.status != "open_eod")
    n_with_tp = sum(1 for t in trades if t.status == "closed_with_tp")
    n_open = sum(1 for t in trades if t.status == "open_eod")
    print()
    print(f"Closed:         {n_closed}  (hit partial TP: {n_with_tp})  P&L ${closed_pnl:+,.2f}")
    print(f"Open MTM:       {n_open}  P&L ${open_pnl:+,.2f}")
    print(f"Skipped at cap: {skipped_at_cap}")
    print(f"Skipped chop:   {skipped_chop}")
    print(f"Circuit broken: {circuit_broken}")
    for d, pnl in by_dir.items():
        print(f"  {d.value:<5} P&L ${pnl:+,.2f}")
    total = closed_pnl + open_pnl
    print(f"Total:          ${total:+,.2f}  ({total/equity*100:+.3f}% of ${equity:,.0f})")


if __name__ == "__main__":
    run_backtest()
