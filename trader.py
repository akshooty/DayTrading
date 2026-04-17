"""Breakout-long live trader.

Runs during regular US market hours. EOD closeout ~2:55 PM CT, then exits.
Launched by launchd at ~8:25 AM CT weekdays.

Set DRY_RUN=1 to log signals without placing orders.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.stream import TradingStream
from dotenv import load_dotenv

from scanner import scan as finviz_scan
from strategy import (
    ArmSignal,
    Bar,
    BreakoutLongState,
    ResetSignal,
    State,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
EOD_EXIT_ET = time(15, 55)

RISK_PCT = 0.005
MIN_STOP_DIST = 0.10
MAX_CONCURRENT = 5
LIMIT_OFFSET = 0.05

log = logging.getLogger("trader")


class BreakoutTrader:
    def __init__(self, dry_run: bool = False):
        load_dotenv()
        self.dry_run = dry_run
        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=True)
        self.data_stream = StockDataStream(key, secret)
        self.trade_stream = TradingStream(key, secret, paper=True)
        self.states: dict[str, BreakoutLongState] = {}
        self.entry_orders: dict[str, str] = {}
        self.open_positions: set[str] = set()
        self.equity: float = 0.0

    def load_watchlist(self) -> list[str]:
        override = os.environ.get("WATCHLIST")
        if override:
            tickers = [s.strip().upper() for s in override.split(",") if s.strip()]
            log.info("watchlist from env (%d): %s", len(tickers), tickers)
            return tickers
        results = finviz_scan()
        tickers = [r["Ticker"] for r in results if r.get("Ticker")]
        log.info("watchlist from finviz (%d): %s", len(tickers), tickers)
        return tickers

    async def on_bar(self, bar) -> None:
        st = self.states.get(bar.symbol)
        if not st:
            return
        internal = Bar(
            open=float(bar.open), high=float(bar.high),
            low=float(bar.low), close=float(bar.close),
            volume=float(bar.volume), timestamp=bar.timestamp,
        )
        sig = st.on_bar(internal)
        log.debug(
            "%s bar h=%.2f l=%.2f state=%s peak=%.2f pb_low=%.2f",
            bar.symbol, bar.high, bar.low, st.state.value,
            st.peak_high, st.pullback_low,
        )
        if isinstance(sig, ArmSignal):
            await self._handle_arm(sig)
        elif isinstance(sig, ResetSignal):
            await self._handle_reset(sig)

    async def _handle_arm(self, s: ArmSignal) -> None:
        busy = len(self.entry_orders) + len(self.open_positions)
        if busy >= MAX_CONCURRENT:
            log.info("%s ARM skipped (at cap %d)", s.symbol, MAX_CONCURRENT)
            self.states[s.symbol].reset_to_watching()
            return

        qty = position_size(self.equity, RISK_PCT, s.entry, s.stop, MIN_STOP_DIST)
        if qty <= 0:
            log.warning("%s ARM skipped — qty 0", s.symbol)
            return

        log.info(
            "%s ARM entry=%.2f stop=%.2f qty=%d risk=$%.2f",
            s.symbol, s.entry, s.stop, qty, qty * max(s.entry - s.stop, MIN_STOP_DIST),
        )
        if self.dry_run:
            return

        req = StopLimitOrderRequest(
            symbol=s.symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            stop_price=round(s.entry, 2),
            limit_price=round(s.entry + LIMIT_OFFSET, 2),
        )
        order = self.trading.submit_order(req)
        self.entry_orders[s.symbol] = str(order.id)

    async def _handle_reset(self, s: ResetSignal) -> None:
        log.info("%s RESET (pullback low breached)", s.symbol)
        oid = self.entry_orders.pop(s.symbol, None)
        if oid and not self.dry_run:
            try:
                self.trading.cancel_order_by_id(oid)
            except Exception as e:
                log.warning("%s cancel failed: %s", s.symbol, e)

    async def on_trade_update(self, msg) -> None:
        order = msg.order
        sym = order.symbol
        st = self.states.get(sym)
        if not st:
            return

        if msg.event == "fill":
            if str(order.side).lower().endswith("buy"):
                filled = int(float(order.filled_qty or 0))
                price = float(order.filled_avg_price or 0)
                log.info("%s ENTRY FILL qty=%d @ %.2f", sym, filled, price)
                st.on_entry_filled()
                self.entry_orders.pop(sym, None)
                self.open_positions.add(sym)
                trail = round(max(price - st.pullback_low, MIN_STOP_DIST), 2)
                if self.dry_run:
                    log.info("%s would trail $%.2f", sym, trail)
                    return
                self.trading.submit_order(TrailingStopOrderRequest(
                    symbol=sym, qty=filled, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY, trail_price=trail,
                ))
            else:
                log.info(
                    "%s EXIT FILL qty=%s @ %s",
                    sym, order.filled_qty, order.filled_avg_price,
                )
                st.on_exit_filled()
                self.open_positions.discard(sym)
        elif msg.event in ("canceled", "expired", "rejected"):
            if str(order.side).lower().endswith("buy"):
                self.entry_orders.pop(sym, None)
                if st.state is State.ARMED:
                    st.reset_to_watching()

    async def eod_closeout(self) -> None:
        log.info("EOD closeout")
        if self.dry_run:
            return
        try:
            self.trading.cancel_orders()
        except Exception as e:
            log.warning("cancel_orders failed: %s", e)
        try:
            self.trading.close_all_positions(cancel_orders=True)
        except Exception as e:
            log.warning("close_all_positions failed: %s", e)

    async def run(self) -> None:
        acct = self.trading.get_account()
        self.equity = float(acct.equity)
        log.info("equity=$%.2f dry_run=%s", self.equity, self.dry_run)

        watchlist = self.load_watchlist()
        if not watchlist:
            log.warning("empty watchlist — exiting")
            return
        for sym in watchlist:
            self.states[sym] = BreakoutLongState(symbol=sym)

        self.data_stream.subscribe_bars(self.on_bar, *watchlist)
        self.trade_stream.subscribe_trade_updates(self.on_trade_update)

        now_et = datetime.now(EASTERN)
        eod_dt = datetime.combine(now_et.date(), EOD_EXIT_ET, tzinfo=EASTERN)
        wait_s = max((eod_dt - now_et).total_seconds(), 0)
        log.info("running until %s (%d s)", eod_dt.isoformat(), int(wait_s))

        data_task = asyncio.create_task(self.data_stream._run_forever())
        trade_task = asyncio.create_task(self.trade_stream._run_forever())
        timer_task = asyncio.create_task(asyncio.sleep(wait_s))

        try:
            done, pending = await asyncio.wait(
                [data_task, trade_task, timer_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            await self.eod_closeout()
            try:
                await self.data_stream.stop_ws()
            except Exception:
                pass
            try:
                await self.trade_stream.stop_ws()
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dry = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    asyncio.run(BreakoutTrader(dry_run=dry).run())


if __name__ == "__main__":
    main()
