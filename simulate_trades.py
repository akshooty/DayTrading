"""Run the strategy backtest for the last N trading days and dump
all round-trip trades to reports/simulated_trades.json in the shape
the dashboard expects.

Usage: FEED=SIP python simulate_trades.py [N]    (default N=10)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

from backtest import run_backtest
from backtest_historical import fetch_daily_bars, movers_from_cache, prior_trading_day


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    today = date.today()
    dates: list[date] = []
    d = today
    while len(dates) < n:
        d = prior_trading_day(d)
        dates.append(d)
    dates.reverse()
    print(f"simulating {n} days: {dates[0]} -> {dates[-1]}", flush=True)

    all_bars = fetch_daily_bars(prior_trading_day(dates[0]), dates[-1])
    trades_out: list[dict] = []

    for day in dates:
        gainers, losers = movers_from_cache(day, all_bars)
        os.environ["LONG_WATCHLIST"] = ",".join(g["Ticker"] for g in gainers)
        os.environ["SHORT_WATCHLIST"] = ",".join(l["Ticker"] for l in losers)
        os.environ["BACKTEST_DATE"] = day.isoformat()
        s = run_backtest(print_report=False)
        print(f"  {day}: {s['n_closed']} closed, {s['n_open']} open, P&L ${s['total_pnl']:+,.2f}", flush=True)

        for t in s["trades"]:
            is_open = t.status == "open_eod"
            trades_out.append({
                "symbol": t.ticker,
                "direction": t.direction.value,
                "qty": t.original_qty,
                "entry_price": t.entry_price,
                "exit_price": None if is_open else t.final_exit_price,
                "entry_at": t.opened_at.isoformat(),
                "exit_at": None if is_open else t.closed_at.isoformat(),
                "pnl": None if is_open else t.pnl,
                "duration_min": None if is_open else (t.closed_at - t.opened_at).total_seconds() / 60,
                "status": "open" if is_open else "closed",
                "strategy": t.strategy,
                "tp_price": t.tp_price,
            })

    os.makedirs("reports", exist_ok=True)
    path = "reports/simulated_trades.json"
    with open(path, "w") as f:
        json.dump(trades_out, f, indent=2, default=str)
    print(f"\nSaved {len(trades_out)} trades to {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
