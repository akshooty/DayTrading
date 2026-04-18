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
    ORBState,
    ResetSignal,
    min_stop_dist,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
MAX_CONCURRENT = 16
MAX_DEPLOYMENT = 2500.0
MAX_RISK_PER_TRADE = 200.0
def _parse_et(env_key: str, default: time) -> time:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    h, m = raw.split(":")
    return time(int(h), int(m))

CHOP_START = _parse_et("CHOP_START_ET", time(11, 30))
CHOP_END = _parse_et("CHOP_END_ET", time(14, 0))
_eod_raw = os.environ.get("EOD_CUTOFF_ET")
EOD_CUTOFF: time | None = _parse_et("EOD_CUTOFF_ET", time(15, 55)) if _eod_raw else None
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
    strategy: str = ""


def _valid(sym: str) -> bool:
    return bool(sym) and sym.isalnum()


def _compute_atr(bars: list, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    recent = bars[-(period + 1):]
    for i in range(1, len(recent)):
        b, prev = recent[i], recent[i - 1]
        tr = max(b.high - b.low, abs(b.high - prev.close), abs(b.low - prev.close))
        trs.append(tr)
    return sum(trs) / period


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


def run_backtest(print_report: bool = True, starting_pnl: float = 0.0) -> dict:
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"].strip(), os.environ["ALPACA_SECRET_KEY"].strip()
    )
    equity = float(os.environ.get("BACKTEST_EQUITY", "953000"))
    daily_loss_cap = -equity * DAILY_LOSS_LIMIT_PCT

    watchlist = load_watchlist()
    longs = [s for s, d in watchlist.items() if d is Direction.LONG]
    shorts = [s for s, d in watchlist.items() if d is Direction.SHORT]
    if print_report:
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
    if print_report:
        print(f"loaded {len(events)} bar events")
        print()

    # Each ticker runs TWO state machines in parallel: the directional
    # breakout (uptrend/pullback/reversal) and ORB (first-10-min range).
    # First one to ARM wins the ticker's position slot.
    states: dict[str, list] = {}
    for sym, d in watchlist.items():
        primary = BreakoutLongState(sym) if d is Direction.LONG else BreakoutShortState(sym)
        states[sym] = [primary, ORBState(sym)]

    positions: dict[str, dict] = {}
    pending: dict[str, tuple[ArmSignal, object]] = {}
    ma9_5m: dict[str, dict] = {}  # per-ticker 5m EMA(9) state
    trades: list[Trade] = []
    locked_out: set[str] = set()  # tickers stopped out >$LOCKOUT_LOSS today; no re-entry
    lockout_loss = float(os.environ.get("LOCKOUT_LOSS", 150))
    skipped_at_cap = 0
    skipped_chop = 0
    skipped_locked = 0
    circuit_broken = False
    realized_pnl = 0.0

    for ts, ticker, b in events:
        sts = states.get(ticker)
        if not sts:
            continue

        # Update 5-min EMA(9) buffer per ticker (used by Phase 2 MA9 exit).
        ma_state = ma9_5m.setdefault(ticker, {"buf": [], "closes": [], "ema": None})
        ma_state["buf"].append(b)
        if ts.astimezone(EASTERN).minute % 5 == 4:
            close_5m = ma_state["buf"][-1].close
            ma_state["buf"] = []
            if ma_state["ema"] is None:
                ma_state["closes"].append(close_5m)
                if len(ma_state["closes"]) >= 9:
                    ma_state["ema"] = sum(ma_state["closes"][-9:]) / 9.0
            else:
                k = 2.0 / (9 + 1)
                ma_state["ema"] = close_5m * k + ma_state["ema"] * (1 - k)

        # EOD force-close: at first bar past cutoff, close any open
        # position at bar.open and skip the rest of management/entries.
        if EOD_CUTOFF is not None and ts.astimezone(EASTERN).time() >= EOD_CUTOFF:
            if ticker in positions:
                pos = positions[ticker]
                d = pos["direction"]
                exit_px = b.open
                final_qty = pos["remaining_qty"]
                final_pnl = (
                    (exit_px - pos["entry"]) * final_qty
                    if d is Direction.LONG
                    else (pos["entry"] - exit_px) * final_qty
                )
                tp_pnl = pos.get("_tp_pnl", 0.0)
                tp_qty = pos.get("_tp_qty", 0)
                realized_pnl += final_pnl
                trades.append(Trade(
                    ticker=ticker, direction=d,
                    entry_price=pos["entry"],
                    original_qty=pos["original_qty"],
                    tp_price=pos["tp_price"], tp_qty=tp_qty,
                    final_exit_price=exit_px,
                    final_exit_qty=final_qty,
                    pnl=tp_pnl + final_pnl,
                    opened_at=pos["opened_at"], closed_at=ts,
                    status="closed_no_tp",
                    strategy=pos.get("strategy", ""),
                ))
                owner = pos.get("owner")
                if owner is not None:
                    owner.on_exit_filled()
                del positions[ticker]
            if ticker in pending:
                del pending[ticker]
            continue

        # Check pending fill
        if ticker in pending:
            sig, armed_state = pending[ticker]
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
                    hybrid_mode = os.environ.get("HYBRID_TP", "1").lower() in ("1", "true", "yes")
                    hybrid_tp_pct = float(os.environ.get("HYBRID_TP_PCT", 4))
                    hybrid_atr_period = int(os.environ.get("HYBRID_ATR_PERIOD", 14))
                    if hybrid_mode:
                        if sig.direction is Direction.LONG:
                            tp_level = fill * (1 + hybrid_tp_pct / 100)
                        else:
                            tp_level = fill * (1 - hybrid_tp_pct / 100)
                        atr_at_entry = _compute_atr(armed_state.bars, hybrid_atr_period)
                    else:
                        if sig.direction is Direction.LONG:
                            tp_level = fill + initial_risk
                        else:
                            tp_level = fill - initial_risk
                        atr_at_entry = 0.0
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
                        owner=armed_state,
                        strategy=type(armed_state).__name__,
                        _atr_at_entry=atr_at_entry,
                        _hybrid_mode=hybrid_mode,
                    )
                    armed_state.on_entry_filled()
                del pending[ticker]

        # Manage open position
        if ticker in positions:
            pos = positions[ticker]
            d = pos["direction"]

            # Alternate exit mode: HARD_TP_PCT — exit full position at
            # entry * (1 + pct/100) for long or (1 - pct/100) for short,
            # no partial TP, no trail. Initial stop still applies.
            # Hard take-profit at fixed % beats 1R-partial-+-trail on
            # 22-day window (+9% P&L). Override via HARD_TP_PCT=0 for
            # the legacy two-phase behavior.
            hard_tp = 0.0 if pos.get("_hybrid_mode") else float(os.environ.get("HARD_TP_PCT") or 7)
            if hard_tp > 0 and pos.get("_hard_tp_target") is None:
                if d is Direction.LONG:
                    pos["_hard_tp_target"] = pos["entry"] * (1 + hard_tp / 100)
                else:
                    pos["_hard_tp_target"] = pos["entry"] * (1 - hard_tp / 100)
                pos["_hard_tp_mode"] = True
            if pos.get("_hard_tp_mode"):
                hit_hard = (
                    (d is Direction.LONG and b.high >= pos["_hard_tp_target"])
                    or (d is Direction.SHORT and b.low <= pos["_hard_tp_target"])
                )
                hit_stop = (
                    (d is Direction.LONG and b.low <= pos["initial_stop"])
                    or (d is Direction.SHORT and b.high >= pos["initial_stop"])
                )
                exit_px = None
                if hit_hard:
                    exit_px = pos["_hard_tp_target"]
                elif hit_stop:
                    exit_px = pos["initial_stop"]
                if exit_px is not None:
                    pnl = (
                        (exit_px - pos["entry"]) * pos["original_qty"]
                        if d is Direction.LONG
                        else (pos["entry"] - exit_px) * pos["original_qty"]
                    )
                    realized_pnl += pnl
                    trades.append(Trade(
                        ticker=ticker, direction=d,
                        entry_price=pos["entry"],
                        original_qty=pos["original_qty"],
                        tp_price=exit_px if hit_hard else None,
                        tp_qty=pos["original_qty"] if hit_hard else 0,
                        final_exit_price=exit_px,
                        final_exit_qty=pos["original_qty"],
                        pnl=pnl,
                        opened_at=pos["opened_at"], closed_at=ts,
                        status="closed_with_tp" if hit_hard else "closed_no_tp",
                        strategy=pos.get("strategy", ""),
                    ))
                    owner = pos.get("owner")
                    if owner is not None:
                        owner.on_exit_filled()
                    if pnl < -lockout_loss:
                        locked_out.add(ticker)
                    del positions[ticker]
                    # Skip the standard two-phase block below
                    continue

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
                use_ma9 = (
                    pos.get("_hybrid_mode")
                    and os.environ.get("EXIT_MA9_5M", "").lower() in ("1", "true", "yes")
                )
                if use_ma9:
                    ema9 = ma9_5m.get(ticker, {}).get("ema")
                    if ema9 is not None:
                        if d is Direction.LONG and b.close < ema9:
                            pos["stop"] = b.close
                        elif d is Direction.SHORT and b.close > ema9:
                            pos["stop"] = b.close
                else:
                    if pos.get("_hybrid_mode"):
                        atr_mult = float(os.environ.get("HYBRID_ATR_MULT", 1.5))
                        trail_dist = max(pos["_atr_at_entry"] * atr_mult, min_stop_dist(pos["entry"]))
                    else:
                        trail_dist = pos["initial_risk"]
                    if d is Direction.LONG:
                        if b.high > pos["extreme"]:
                            pos["extreme"] = b.high
                            pos["stop"] = max(pos["stop"], pos["extreme"] - trail_dist)
                    else:
                        if b.low < pos["extreme"]:
                            pos["extreme"] = b.low
                            pos["stop"] = min(pos["stop"], pos["extreme"] + trail_dist)

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
                    strategy=pos.get("strategy", ""),
                ))
                owner = pos.get("owner")
                if owner is not None:
                    owner.on_exit_filled()
                if (tp_pnl + final_pnl) < -lockout_loss:
                    locked_out.add(ticker)
                del positions[ticker]

        # Circuit breaker — stop taking new entries
        if realized_pnl <= daily_loss_cap and not circuit_broken:
            circuit_broken = True

        # Drive ALL state machines for this ticker
        ib = IntBar(
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=b.volume, timestamp=ts,
        )
        for st in sts:
            sig = st.on_bar(ib)
            if isinstance(sig, ArmSignal):
                if circuit_broken:
                    st.reset_to_watching()
                    continue
                if _in_chop(ts):
                    skipped_chop += 1
                    st.reset_to_watching()
                    continue
                if ticker in locked_out:
                    skipped_locked += 1
                    st.reset_to_watching()
                    continue
                if ticker in positions or ticker in pending:
                    # Another state machine already owns this ticker
                    st.reset_to_watching()
                    continue
                scale = os.environ.get("SCALE_CONCURRENT", "").lower() in ("1", "true", "yes")
                step = float(os.environ.get("SCALE_CONCURRENT_STEP", 2500))
                cum_pnl = starting_pnl + realized_pnl
                dynamic_max = MAX_CONCURRENT + (max(0, int(cum_pnl // step)) if scale else 0)
                if len(positions) + len(pending) >= dynamic_max:
                    skipped_at_cap += 1
                    st.reset_to_watching()
                    continue
                pending[ticker] = (sig, st)
            elif isinstance(sig, ResetSignal):
                if ticker in pending and pending[ticker][1] is st:
                    del pending[ticker]

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
            strategy=pos.get("strategy", ""),
        ))

    # Report
    closed_pnl = sum(t.pnl for t in trades if t.status != "open_eod")
    open_pnl = sum(t.pnl for t in trades if t.status == "open_eod")
    by_dir: dict = {}
    for t in trades:
        by_dir.setdefault(t.direction.value, 0.0)
        by_dir[t.direction.value] += t.pnl
    n_closed = sum(1 for t in trades if t.status != "open_eod")
    n_with_tp = sum(1 for t in trades if t.status == "closed_with_tp")
    n_open = sum(1 for t in trades if t.status == "open_eod")

    if print_report:
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
        print()
        print(f"Closed:         {n_closed}  (hit partial TP: {n_with_tp})  P&L ${closed_pnl:+,.2f}")
        print(f"Open MTM:       {n_open}  P&L ${open_pnl:+,.2f}")
        print(f"Skipped at cap: {skipped_at_cap}")
        print(f"Skipped chop:   {skipped_chop}")
        print(f"Skipped locked: {skipped_locked}")
        print(f"Circuit broken: {circuit_broken}")
        for d, pnl in by_dir.items():
            print(f"  {d:<5} P&L ${pnl:+,.2f}")
        total = closed_pnl + open_pnl
        print(f"Total:          ${total:+,.2f}  ({total/equity*100:+.3f}% of ${equity:,.0f})")

    return {
        "equity": equity,
        "tickers": len(watchlist),
        "longs": len(longs),
        "shorts": len(shorts),
        "events": len(events),
        "n_closed": n_closed,
        "n_open": n_open,
        "n_tp": n_with_tp,
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "total_pnl": closed_pnl + open_pnl,
        "by_dir": by_dir,
        "skipped_at_cap": skipped_at_cap,
        "skipped_chop": skipped_chop,
        "skipped_locked": skipped_locked,
        "circuit_broken": circuit_broken,
        "trades": trades,
    }


if __name__ == "__main__":
    run_backtest()
