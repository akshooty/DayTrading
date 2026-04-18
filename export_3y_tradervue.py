"""Run backtest across last N trading days and export every trade to
a TraderVue Generic CSV (Date,Time,Symbol,Side,Quantity,Price,Commission).

Usage: FEED=SIP python export_3y_tradervue.py [N] [out_path]
       defaults: N=760 (~3yrs), out=reports/trades_3y_tradervue.csv
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date
from zoneinfo import ZoneInfo

from backtest import run_backtest
from backtest_historical import (
    fetch_daily_bars, movers_from_cache, prior_trading_day,
)
from strategy import Direction

CST = ZoneInfo("America/Chicago")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 760
    out_path = sys.argv[2] if len(sys.argv) > 2 else "reports/trades_3y_tradervue.csv"

    today = date.today()
    dates: list[date] = []
    d = today
    while len(dates) < n:
        d = prior_trading_day(d)
        dates.append(d)
    dates.reverse()

    print(f"export {n} trading days: {dates[0]} -> {dates[-1]}", flush=True)
    range_start = prior_trading_day(dates[0])
    print(f"prefetching daily bars {range_start} -> {dates[-1]}...", flush=True)
    all_bars = fetch_daily_bars(range_start, dates[-1])
    print(flush=True)

    cum_pnl = 0.0
    all_trades = []
    for i, day in enumerate(dates):
        gainers, losers = movers_from_cache(day, all_bars)
        os.environ["LONG_WATCHLIST"] = ",".join(g["Ticker"] for g in gainers)
        os.environ["SHORT_WATCHLIST"] = ",".join(l["Ticker"] for l in losers)
        os.environ["BACKTEST_DATE"] = day.isoformat()
        try:
            s = run_backtest(print_report=False, starting_pnl=cum_pnl)
            cum_pnl += s["total_pnl"]
            all_trades.extend(s.get("trades", []))
        except Exception as e:
            print(f"  {day} failed: {e}", flush=True)
        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            print(
                f"  {i+1}/{n} ({day}) {len(all_trades)} trades, cum=${cum_pnl:+,.2f}",
                flush=True,
            )

    print(f"\nwriting {len(all_trades)} trades to {out_path}", flush=True)
    rows = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Time", "Symbol", "Side", "Quantity", "Price", "Commission"])
        for t in all_trades:
            is_long = t.direction is Direction.LONG
            entry_side = "B" if is_long else "S"
            exit_side = "S" if is_long else "B"
            entry_dt = t.opened_at.astimezone(CST)
            close_dt = t.closed_at.astimezone(CST)
            w.writerow([
                entry_dt.date().isoformat(),
                entry_dt.time().strftime("%H:%M:%S"),
                t.ticker, entry_side, t.original_qty,
                f"{t.entry_price:.4f}", "0",
            ])
            rows += 1
            if t.tp_price is not None and t.tp_qty > 0:
                w.writerow([
                    close_dt.date().isoformat(),
                    close_dt.time().strftime("%H:%M:%S"),
                    t.ticker, exit_side, t.tp_qty,
                    f"{t.tp_price:.4f}", "0",
                ])
                rows += 1
            if t.final_exit_qty > 0:
                w.writerow([
                    close_dt.date().isoformat(),
                    close_dt.time().strftime("%H:%M:%S"),
                    t.ticker, exit_side, t.final_exit_qty,
                    f"{t.final_exit_price:.4f}", "0",
                ])
                rows += 1
    print(f"wrote {rows} CSV rows. cum P&L: ${cum_pnl:+,.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
