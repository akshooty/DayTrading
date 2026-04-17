"""Run the historical backtest across the last N trading days and
aggregate results.

Usage: FEED=SIP python backtest_week.py [N]   (default N=5)
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from backtest import run_backtest
from backtest_historical import find_movers, prior_trading_day


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    today = date.today()
    dates: list[date] = []
    d = today
    while len(dates) < n:
        d = prior_trading_day(d)
        dates.append(d)
    dates.reverse()

    print(f"backtesting {n} trading days: {dates[0]} -> {dates[-1]}")
    print()

    summaries = []
    for day in dates:
        print(f"=== {day} ({day.strftime('%A')}) ===")
        gainers, losers = find_movers(day)
        print(f"universe: {len(gainers)} gainers + {len(losers)} losers")

        longs = ",".join(g["Ticker"] for g in gainers)
        shorts = ",".join(l["Ticker"] for l in losers)

        os.environ["LONG_WATCHLIST"] = longs
        os.environ["SHORT_WATCHLIST"] = shorts
        os.environ["BACKTEST_DATE"] = day.isoformat()

        try:
            s = run_backtest(print_report=False)
        except Exception as e:
            print(f"  failed: {e}")
            continue

        s["date"] = day
        summaries.append(s)
        print(
            f"  {s['n_closed']} closed ({s['n_tp']} TP), {s['n_open']} open, "
            f"skipped {s['skipped_at_cap']}@cap {s['skipped_chop']}@chop, "
            f"P&L ${s['total_pnl']:+,.2f}"
        )
        print()

    # Aggregate
    if not summaries:
        print("no days succeeded")
        return 1
    print()
    print("=" * 75)
    print(f"{'Date':<12} {'Long':>11} {'Short':>11} {'Closed':>6} {'TP':>4} {'Skip':>5} {'P&L':>12}")
    print("-" * 75)
    total = 0.0
    total_long = 0.0
    total_short = 0.0
    total_closed = 0
    total_tp = 0
    total_skipped = 0
    winning_days = 0
    for s in summaries:
        long_pnl = s["by_dir"].get("long", 0.0)
        short_pnl = s["by_dir"].get("short", 0.0)
        total += s["total_pnl"]
        total_long += long_pnl
        total_short += short_pnl
        total_closed += s["n_closed"]
        total_tp += s["n_tp"]
        total_skipped += s["skipped_at_cap"] + s["skipped_chop"]
        if s["total_pnl"] > 0:
            winning_days += 1
        print(
            f"{s['date']!s:<12} ${long_pnl:>+9.2f}  ${short_pnl:>+9.2f}  "
            f"{s['n_closed']:>6} {s['n_tp']:>4} "
            f"{s['skipped_at_cap']+s['skipped_chop']:>5} ${s['total_pnl']:>+10.2f}"
        )
    print("-" * 75)
    print(
        f"{'TOTAL':<12} ${total_long:>+9.2f}  ${total_short:>+9.2f}  "
        f"{total_closed:>6} {total_tp:>4} {total_skipped:>5} ${total:>+10.2f}"
    )
    print()
    equity = summaries[0]["equity"]
    print(f"Equity base: ${equity:,.0f}")
    print(f"Total P&L:   ${total:+,.2f}  ({total/equity*100:+.3f}% of equity)")
    print(f"Days:        {len(summaries)}  ({winning_days} winners, {len(summaries)-winning_days} losers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
