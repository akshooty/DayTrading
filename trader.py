"""Breakout-long + breakout-short live trader.

Subscribes to Alpaca 1-min bars for today's gainers (LONG candidates)
and losers (SHORT candidates), drives per-symbol state machines, and
submits stop-limit entries + trailing-stop exits on Alpaca paper.

Runs during regular US session; EOD closeout at 2:55 PM CT.
Launched by launchd at 8:25 AM CT weekdays.

Env flags:
  DRY_RUN=1     disable order submission (log only)
  WATCHLIST=... comma-separated long-only ticker override for testing
  LOG_LEVEL=... DEBUG|INFO (default INFO)
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

from scanner import scan_gainers, scan_losers
from strategy import (
    ArmSignal,
    Bar,
    BreakoutLongState,
    BreakoutShortState,
    BreakoutState,
    Direction,
    ResetSignal,
    State,
    min_stop_dist,
    position_size,
)

EASTERN = ZoneInfo("America/New_York")
EOD_EXIT_ET = time(15, 55)
CHOP_START = time(11, 30)
CHOP_END = time(14, 0)

RISK_PCT = 0.005
MAX_CONCURRENT = 8
MAX_DEPLOYMENT = 5000.0
LIMIT_OFFSET = 0.05
DAILY_LOSS_LIMIT_PCT = 0.01

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
        self.states: dict[str, BreakoutState] = {}
        self.entry_orders: dict[str, str] = {}
        self.open_positions: set[str] = set()
        self.entry_prices: dict[str, float] = {}
        self.equity: float = 0.0
        self.realized_pnl: float = 0.0
        self.circuit_broken: bool = False

    def load_watchlist(self) -> dict[str, Direction]:
        def valid(s: str) -> bool:
            return bool(s) and s.isalnum()

        override = os.environ.get("WATCHLIST")
        if override:
            syms = {s.strip().upper(): Direction.LONG for s in override.split(",") if valid(s.strip().upper())}
            log.info("watchlist from env (%d, all LONG): %s", len(syms), list(syms))
            return syms
        gainers = [r["Ticker"] for r in scan_gainers() if valid(r.get("Ticker") or "")]
        losers = [r["Ticker"] for r in scan_losers() if valid(r.get("Ticker") or "")]
        out: dict[str, Direction] = {t: Direction.LONG for t in gainers}
        for t in losers:
            if t not in out:
                out[t] = Direction.SHORT
        log.info(
            "watchlist: %d gainers (LONG) + %d losers (SHORT) = %d tickers",
            len(gainers), len(losers), len(out),
        )
        return out

    async def on_bar(self, bar) -> None:
        st = self.states.get(bar.symbol)
        if not st:
            return
        ib = Bar(
            open=float(bar.open), high=float(bar.high),
            low=float(bar.low), close=float(bar.close),
            volume=float(bar.volume), timestamp=bar.timestamp,
        )
        sig = st.on_bar(ib)
        log.debug(
            "%s dir=%s bar h=%.2f l=%.2f state=%s",
            bar.symbol, st.direction.value, bar.high, bar.low, st.state.value,
        )
        if isinstance(sig, ArmSignal):
            await self._handle_arm(sig)
        elif isinstance(sig, ResetSignal):
            await self._handle_reset(sig)

    def _in_chop_now(self) -> bool:
        t = datetime.now(EASTERN).time()
        return CHOP_START <= t < CHOP_END

    async def _handle_arm(self, s: ArmSignal) -> None:
        if self.circuit_broken:
            log.info("%s ARM skipped (circuit breaker)", s.symbol)
            self.states[s.symbol].reset_to_watching()
            return
        if self._in_chop_now():
            log.info("%s ARM skipped (11:30-14:00 ET chop window)", s.symbol)
            self.states[s.symbol].reset_to_watching()
            return
        busy = len(self.entry_orders) + len(self.open_positions)
        if busy >= MAX_CONCURRENT:
            log.info("%s ARM skipped (at cap %d)", s.symbol, MAX_CONCURRENT)
            self.states[s.symbol].reset_to_watching()
            return

        qty = position_size(
            self.equity, RISK_PCT, s.entry, s.stop,
            max_deployment=MAX_DEPLOYMENT,
        )
        if qty <= 0:
            log.warning("%s ARM skipped — qty 0", s.symbol)
            return

        log.info(
            "%s ARM dir=%s entry=%.2f stop=%.2f qty=%d deploy=$%.2f",
            s.symbol, s.direction.value, s.entry, s.stop, qty, qty * s.entry,
        )

        if self.dry_run:
            return

        if s.direction is Direction.LONG:
            req = StopLimitOrderRequest(
                symbol=s.symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                stop_price=round(s.entry, 2),
                limit_price=round(s.entry + LIMIT_OFFSET, 2),
            )
        else:
            try:
                asset = self.trading.get_asset(s.symbol)
                if not getattr(asset, "shortable", False):
                    log.info("%s not shortable — skipping", s.symbol)
                    self.states[s.symbol].reset_to_watching()
                    return
            except Exception as e:
                log.warning("%s asset lookup failed: %s", s.symbol, e)
                return
            req = StopLimitOrderRequest(
                symbol=s.symbol, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(s.entry, 2),
                limit_price=round(s.entry - LIMIT_OFFSET, 2),
            )

        order = self.trading.submit_order(req)
        self.entry_orders[s.symbol] = str(order.id)

    async def _handle_reset(self, s: ResetSignal) -> None:
        log.info("%s RESET dir=%s (setup invalidated)", s.symbol, s.direction.value)
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

        is_entry = str(order.id) == self.entry_orders.get(sym)

        if msg.event == "fill":
            if is_entry:
                filled = int(float(order.filled_qty or 0))
                price = float(order.filled_avg_price or 0)
                log.info(
                    "%s ENTRY FILL dir=%s qty=%d @ %.2f",
                    sym, st.direction.value, filled, price,
                )
                st.on_entry_filled()
                self.entry_orders.pop(sym, None)
                self.open_positions.add(sym)
                self.entry_prices[sym] = price

                msd = min_stop_dist(price)
                if st.direction is Direction.LONG:
                    assert isinstance(st, BreakoutLongState)
                    trail = round(max(price - st.pullback_low, msd), 2)
                    exit_side = OrderSide.SELL
                else:
                    assert isinstance(st, BreakoutShortState)
                    trail = round(max(st.bounce_high - price, msd), 2)
                    exit_side = OrderSide.BUY

                if self.dry_run:
                    log.info("%s (dry) trail $%.2f exit=%s", sym, trail, exit_side.value)
                    return
                self.trading.submit_order(TrailingStopOrderRequest(
                    symbol=sym, qty=filled, side=exit_side,
                    time_in_force=TimeInForce.DAY, trail_price=trail,
                ))
            else:
                exit_qty = int(float(order.filled_qty or 0))
                exit_price = float(order.filled_avg_price or 0)
                entry_price = self.entry_prices.pop(sym, 0.0)
                if entry_price:
                    trade_pnl = (
                        (exit_price - entry_price) * exit_qty
                        if st.direction is Direction.LONG
                        else (entry_price - exit_price) * exit_qty
                    )
                    self.realized_pnl += trade_pnl
                    if (
                        self.realized_pnl <= -self.equity * DAILY_LOSS_LIMIT_PCT
                        and not self.circuit_broken
                    ):
                        self.circuit_broken = True
                        log.warning(
                            "CIRCUIT BREAKER TRIPPED — realized P&L $%.2f <= -%.1f%% equity. No new entries.",
                            self.realized_pnl, DAILY_LOSS_LIMIT_PCT * 100,
                        )
                log.info(
                    "%s EXIT FILL dir=%s qty=%d @ %.2f realized=$%.2f",
                    sym, st.direction.value, exit_qty, exit_price, self.realized_pnl,
                )
                st.on_exit_filled()
                self.open_positions.discard(sym)
        elif msg.event in ("canceled", "expired", "rejected"):
            if is_entry:
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
        for sym, direction in watchlist.items():
            if direction is Direction.LONG:
                self.states[sym] = BreakoutLongState(symbol=sym)
            else:
                self.states[sym] = BreakoutShortState(symbol=sym)

        self.data_stream.subscribe_bars(self.on_bar, *watchlist.keys())
        self.trade_stream.subscribe_trade_updates(self.on_trade_update)

        now_et = datetime.now(EASTERN)
        eod_dt = datetime.combine(now_et.date(), EOD_EXIT_ET, tzinfo=EASTERN)
        wait_s = max((eod_dt - now_et).total_seconds(), 0)
        log.info("running until %s (%d s)", eod_dt.isoformat(), int(wait_s))

        data_task = asyncio.create_task(self.data_stream._run_forever())
        trade_task = asyncio.create_task(self.trade_stream._run_forever())
        timer_task = asyncio.create_task(asyncio.sleep(wait_s))

        try:
            _, pending = await asyncio.wait(
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
