"""Breakout-long + breakout-short live trader.

Subscribes to Alpaca 1-min bars for today's gainers/losers, drives
per-symbol state machines, and submits hybrid TP + ATR-trail exits
on Alpaca paper. Launched by launchd at 8:25 AM CT weekdays.

Hybrid exit (matches backtest):
  Phase 1: Static stop on full qty + take-profit limit at +HYBRID_TP_PCT
           on half qty.
  Phase 2 (after TP1 fills): trailing stop on remaining half with
           trail = ATR(at entry) * HYBRID_ATR_MULT.

Env flags:
  DRY_RUN=1            disable order submission (log only)
  WATCHLIST=...        long-only ticker override for testing
  LOG_LEVEL=...        DEBUG|INFO (default INFO)
  LOCKOUT_LOSS         re-entry blocked if loss > $X (default 0 => any loss)
  HYBRID_TP_PCT        Phase 1 take-profit % (4)
  HYBRID_ATR_PERIOD    bars used for ATR (14)
  HYBRID_ATR_MULT      Phase 2 trail = ATR * mult (1.5)
  SHORT_SIZE_MULT      qty multiplier on shorts (1.0)
  MEME_SIZE_MULT       qty multiplier on leveraged ETFs/meme (0.5)
  REGIME_SIZE_MULT     qty multiplier when SPY gap > REGIME_GAP_PCT (0.5)
  REGIME_GAP_PCT       SPY |today_open vs prior_close| % to trip regime (1.5)
  BLOCKED_TICKERS      comma-list of tickers to never trade (RPAY)
  GMAIL_USER           sender Gmail address for trade alerts
  GMAIL_APP_PASSWORD   16-char Gmail app password
  EMAIL_TO             recipient for trade alerts (default: GMAIL_USER)
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
from datetime import datetime, time, timedelta
from email.message import EmailMessage
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
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

MAX_CONCURRENT = 16
MAX_DEPLOYMENT = 2500.0
MAX_RISK_PER_TRADE = 200.0
LIMIT_OFFSET = 0.05
DAILY_LOSS_LIMIT_PCT = 0.01

LOCKOUT_LOSS = float(os.environ.get("LOCKOUT_LOSS", 0))  # 0 => any losing close locks ticker
HYBRID_TP_PCT = float(os.environ.get("HYBRID_TP_PCT", 4))
HYBRID_ATR_PERIOD = int(os.environ.get("HYBRID_ATR_PERIOD", 14))
HYBRID_ATR_MULT = float(os.environ.get("HYBRID_ATR_MULT", 1.5))

# Tradervue-log insights (mirrored from backtest.py).
SHORT_SIZE_MULT = float(os.environ.get("SHORT_SIZE_MULT", 1.0))
MEME_SIZE_MULT = float(os.environ.get("MEME_SIZE_MULT", 0.5))
REGIME_SIZE_MULT = float(os.environ.get("REGIME_SIZE_MULT", 0.5))
REGIME_GAP_PCT = float(os.environ.get("REGIME_GAP_PCT", 1.5)) / 100.0
MEME_LEVERAGED_TICKERS = {
    "SOXL", "SOXS", "TSLL", "TSLZ", "MSTU", "MSTZ", "UVIX", "UVXY",
    "OPEN", "BBAI", "LCID", "PLUG",
    "SQQQ", "TQQQ", "SPXL", "SPXS", "FNGU", "FNGD", "NVDL", "NVDS",
}
BLOCKED_TICKERS = {
    s.strip().upper() for s in os.environ.get("BLOCKED_TICKERS", "RPAY").split(",") if s.strip()
}

log = logging.getLogger("trader")


def _send_email_sync(subject: str, body: str) -> None:
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("EMAIL_TO", user)
    if not (user and pw and to):
        return
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)


async def _email(subject: str, body: str) -> None:
    try:
        await asyncio.to_thread(_send_email_sync, subject, body)
    except Exception as e:
        log.warning("email send failed: %s", e)


def _compute_atr(bars: list[Bar], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    recent = bars[-(period + 1):]
    for i in range(1, len(recent)):
        b, prev = recent[i], recent[i - 1]
        tr = max(b.high - b.low, abs(b.high - prev.close), abs(b.low - prev.close))
        trs.append(tr)
    return sum(trs) / period


class BreakoutTrader:
    def __init__(self, dry_run: bool = False):
        load_dotenv()
        self.dry_run = dry_run
        key = os.environ["ALPACA_API_KEY"].strip()
        secret = os.environ["ALPACA_SECRET_KEY"].strip()
        self.trading = TradingClient(key, secret, paper=True)
        self.data_stream = StockDataStream(key, secret)
        self.trade_stream = TradingStream(key, secret, paper=True)
        self.data_client = StockHistoricalDataClient(key, secret)
        self.states: dict[str, BreakoutState] = {}
        self.entry_orders: dict[str, str] = {}
        self.positions: dict[str, dict] = {}
        self.locked_out: set[str] = set()
        self.equity: float = 0.0
        self.realized_pnl: float = 0.0
        self.circuit_broken: bool = False
        self.regime_reduced: bool = False

    def _check_regime(self) -> bool:
        """Volatile session if SPY |latest vs prior_close| > REGIME_GAP_PCT.

        Runs at startup (~9:25 ET, 5 min before open). Latest trade is
        the most recent pre-market print; falls back to free-tier feed
        if SIP not available. Any failure => proceed at full size.
        """
        try:
            today = datetime.now(EASTERN).date()
            req = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Day,
                start=datetime.combine(today - timedelta(days=10), time(0, 0)),
                end=datetime.combine(today, time(0, 0)),
            )
            bars = self.data_client.get_stock_bars(req).data.get("SPY", [])
            prior = next((b for b in reversed(bars) if b.timestamp.date() < today), None)
            if not prior or prior.close <= 0:
                return False
            trade = self.data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=["SPY"])
            ).get("SPY")
            if trade is None:
                return False
            current = float(trade.price)
            gap = abs(current - float(prior.close)) / float(prior.close)
            log.info(
                "regime: SPY prior_close=%.2f latest=%.2f gap=%.2f%% (threshold %.2f%%)",
                prior.close, current, gap * 100, REGIME_GAP_PCT * 100,
            )
            return gap > REGIME_GAP_PCT
        except Exception as e:
            log.warning("regime check failed (%s) — proceeding at full size", e)
            return False

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
        if s.symbol in BLOCKED_TICKERS:
            log.info("%s ARM skipped (blocked ticker)", s.symbol)
            self.states[s.symbol].reset_to_watching()
            return
        if self._in_chop_now():
            log.info("%s ARM skipped (11:30-14:00 ET chop window)", s.symbol)
            self.states[s.symbol].reset_to_watching()
            return
        if s.symbol in self.locked_out:
            log.info("%s ARM skipped (locked out — prior losing close)", s.symbol)
            self.states[s.symbol].reset_to_watching()
            return
        busy = len(self.entry_orders) + len(self.positions)
        if busy >= MAX_CONCURRENT:
            log.info("%s ARM skipped (at cap %d)", s.symbol, MAX_CONCURRENT)
            self.states[s.symbol].reset_to_watching()
            return

        qty = position_size(
            s.entry, s.stop,
            max_deployment=MAX_DEPLOYMENT,
            max_risk=MAX_RISK_PER_TRADE,
        )
        size_factors = []
        if s.direction is Direction.SHORT and SHORT_SIZE_MULT != 1.0:
            qty = int(qty * SHORT_SIZE_MULT)
            size_factors.append(f"short x{SHORT_SIZE_MULT}")
        if s.symbol in MEME_LEVERAGED_TICKERS:
            qty = int(qty * MEME_SIZE_MULT)
            size_factors.append(f"meme x{MEME_SIZE_MULT}")
        if self.regime_reduced:
            qty = int(qty * REGIME_SIZE_MULT)
            size_factors.append(f"regime x{REGIME_SIZE_MULT}")
        if qty < 2:
            log.warning("%s ARM skipped — qty %d (need >= 2 for hybrid TP)", s.symbol, qty)
            self.states[s.symbol].reset_to_watching()
            return
        if size_factors:
            log.info("%s size adjustments: %s -> qty=%d", s.symbol, ", ".join(size_factors), qty)

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
        oid = str(order.id)
        st = self.states.get(sym)
        if not st:
            return

        if msg.event == "fill":
            if oid == self.entry_orders.get(sym):
                await self._handle_entry_fill(st, order)
                return
            pos = self.positions.get(sym)
            if pos is None:
                return
            if oid == pos.get("tp_order_id"):
                await self._handle_tp1_fill(st, order, pos)
            elif oid == pos.get("stop_order_id"):
                await self._handle_exit_fill(st, order, pos)
        elif msg.event in ("canceled", "expired", "rejected"):
            if oid == self.entry_orders.get(sym):
                self.entry_orders.pop(sym, None)
                if st.state is State.ARMED:
                    st.reset_to_watching()

    async def _handle_entry_fill(self, st: BreakoutState, order) -> None:
        sym = order.symbol
        filled = int(float(order.filled_qty or 0))
        price = float(order.filled_avg_price or 0)
        log.info(
            "%s ENTRY FILL dir=%s qty=%d @ %.2f",
            sym, st.direction.value, filled, price,
        )
        st.on_entry_filled()
        self.entry_orders.pop(sym, None)

        exit_side = OrderSide.SELL if st.direction is Direction.LONG else OrderSide.BUY

        if filled < 2:
            log.warning("%s entry filled qty=%d (<2) — closing immediately", sym, filled)
            if filled > 0 and not self.dry_run:
                try:
                    self.trading.submit_order(MarketOrderRequest(
                        symbol=sym, qty=filled, side=exit_side,
                        time_in_force=TimeInForce.DAY,
                    ))
                except Exception as e:
                    log.error("%s emergency close failed: %s", sym, e)
            return

        atr = _compute_atr(st.bars, HYBRID_ATR_PERIOD)
        half = filled // 2
        if st.direction is Direction.LONG:
            assert isinstance(st, BreakoutLongState)
            tp_level = round(price * (1 + HYBRID_TP_PCT / 100), 2)
            initial_stop = round(st.pullback_low, 2)
        else:
            assert isinstance(st, BreakoutShortState)
            tp_level = round(price * (1 - HYBRID_TP_PCT / 100), 2)
            initial_stop = round(st.bounce_high, 2)

        pos = {
            "direction": st.direction,
            "entry": price,
            "original_qty": filled,
            "remaining_qty": filled,
            "tp_level": tp_level,
            "initial_stop": initial_stop,
            "atr_at_entry": atr,
            "phase": 1,
            "tp_pnl": 0.0,
            "stop_order_id": None,
            "tp_order_id": None,
        }
        self.positions[sym] = pos

        if self.dry_run:
            log.info(
                "%s (dry) Phase1 stop=%.2f tp=%.2f atr=%.2f half=%d",
                sym, initial_stop, tp_level, atr, half,
            )
            return

        try:
            stop_order = self.trading.submit_order(StopOrderRequest(
                symbol=sym, qty=filled, side=exit_side,
                time_in_force=TimeInForce.DAY,
                stop_price=initial_stop,
            ))
            pos["stop_order_id"] = str(stop_order.id)
            tp_order = self.trading.submit_order(LimitOrderRequest(
                symbol=sym, qty=half, side=exit_side,
                time_in_force=TimeInForce.DAY,
                limit_price=tp_level,
            ))
            pos["tp_order_id"] = str(tp_order.id)
            log.info(
                "%s Phase1 orders: stop=%.2f (qty=%d), tp=%.2f (qty=%d), atr=%.2f",
                sym, initial_stop, filled, tp_level, half, atr,
            )
        except Exception as e:
            log.error("%s failed to submit Phase1 orders: %s", sym, e)

        await _email(
            f"[Trader] ENTRY {sym} {st.direction.value.upper()} {filled}@${price:.2f}",
            (
                f"Symbol:     {sym}\n"
                f"Direction:  {st.direction.value}\n"
                f"Qty:        {filled}\n"
                f"Entry:      ${price:.2f}\n"
                f"Stop:       ${initial_stop:.2f}\n"
                f"TP1 (half): ${tp_level:.2f} ({HYBRID_TP_PCT:.1f}%)\n"
                f"ATR trail:  ${atr * HYBRID_ATR_MULT:.2f} (after TP1)\n"
                f"Deployment: ${filled * price:,.2f}\n"
                f"Day P&L:    ${self.realized_pnl:+,.2f}\n"
                f"Time:       {datetime.now(EASTERN).isoformat()}\n"
            ),
        )

    async def _handle_tp1_fill(self, st: BreakoutState, order, pos: dict) -> None:
        sym = order.symbol
        filled = int(float(order.filled_qty or 0))
        price = float(order.filled_avg_price or 0)
        d = pos["direction"]
        tp_pnl = (
            (price - pos["entry"]) * filled if d is Direction.LONG
            else (pos["entry"] - price) * filled
        )
        pos["tp_pnl"] += tp_pnl
        pos["remaining_qty"] -= filled
        pos["phase"] = 2
        pos["tp_order_id"] = None
        self.realized_pnl += tp_pnl
        log.info(
            "%s TP1 FILL qty=%d @ %.2f pnl=$%.2f day_pnl=$%.2f",
            sym, filled, price, tp_pnl, self.realized_pnl,
        )

        if pos["remaining_qty"] <= 0 or self.dry_run:
            if pos["remaining_qty"] <= 0:
                st.on_exit_filled()
                self.positions.pop(sym, None)
            return

        # Cancel static stop covering full qty; submit trailing stop on remainder.
        old_stop_id = pos.get("stop_order_id")
        pos["stop_order_id"] = None
        if old_stop_id:
            try:
                self.trading.cancel_order_by_id(old_stop_id)
            except Exception as e:
                log.warning("%s cancel stop failed: %s", sym, e)

        trail = round(max(pos["atr_at_entry"] * HYBRID_ATR_MULT, min_stop_dist(pos["entry"])), 2)
        exit_side = OrderSide.SELL if d is Direction.LONG else OrderSide.BUY
        try:
            trail_order = self.trading.submit_order(TrailingStopOrderRequest(
                symbol=sym, qty=pos["remaining_qty"], side=exit_side,
                time_in_force=TimeInForce.DAY, trail_price=trail,
            ))
            pos["stop_order_id"] = str(trail_order.id)
            log.info(
                "%s Phase2 trail=$%.2f qty=%d (atr=%.2f mult=%.2f)",
                sym, trail, pos["remaining_qty"], pos["atr_at_entry"], HYBRID_ATR_MULT,
            )
        except Exception as e:
            log.error("%s failed to submit trail: %s", sym, e)

        await _email(
            f"[Trader] TP1 {sym} {filled}@${price:.2f} +${tp_pnl:.2f}",
            (
                f"Symbol:     {sym}\n"
                f"Direction:  {d.value}\n"
                f"Half qty:   {filled} @ ${price:.2f}\n"
                f"Entry:      ${pos['entry']:.2f}\n"
                f"TP1 P&L:    ${tp_pnl:+,.2f}\n"
                f"Remaining:  {pos['remaining_qty']} (now trailing ${trail:.2f})\n"
                f"Day P&L:    ${self.realized_pnl:+,.2f}\n"
                f"Time:       {datetime.now(EASTERN).isoformat()}\n"
            ),
        )

    async def _handle_exit_fill(self, st: BreakoutState, order, pos: dict) -> None:
        sym = order.symbol
        filled = int(float(order.filled_qty or 0))
        price = float(order.filled_avg_price or 0)
        d = pos["direction"]
        exit_pnl = (
            (price - pos["entry"]) * filled if d is Direction.LONG
            else (pos["entry"] - price) * filled
        )
        total_trade_pnl = pos.get("tp_pnl", 0.0) + exit_pnl
        self.realized_pnl += exit_pnl

        # If protective stop fired in Phase 1, cancel the still-open TP limit.
        if pos.get("phase") == 1:
            tp_id = pos.get("tp_order_id")
            if tp_id and not self.dry_run:
                try:
                    self.trading.cancel_order_by_id(tp_id)
                except Exception as e:
                    log.warning("%s cancel tp failed: %s", sym, e)

        log.info(
            "%s EXIT FILL phase=%d qty=%d @ %.2f trade_pnl=$%.2f day_pnl=$%.2f",
            sym, pos.get("phase", 0), filled, price, total_trade_pnl, self.realized_pnl,
        )

        if total_trade_pnl < -LOCKOUT_LOSS:
            self.locked_out.add(sym)
            log.info(
                "%s locked out (trade pnl $%+.2f, threshold $-%.0f)",
                sym, total_trade_pnl, LOCKOUT_LOSS,
            )

        if (
            self.realized_pnl <= -self.equity * DAILY_LOSS_LIMIT_PCT
            and not self.circuit_broken
        ):
            self.circuit_broken = True
            log.warning(
                "CIRCUIT BREAKER TRIPPED — realized P&L $%.2f <= -%.1f%% equity. No new entries.",
                self.realized_pnl, DAILY_LOSS_LIMIT_PCT * 100,
            )

        st.on_exit_filled()
        self.positions.pop(sym, None)

        sign = "WIN" if total_trade_pnl >= 0 else "LOSS"
        await _email(
            f"[Trader] EXIT {sym} {sign} ${total_trade_pnl:+.2f}",
            (
                f"Symbol:     {sym}\n"
                f"Direction:  {d.value}\n"
                f"Phase:      {pos.get('phase', 0)} ({'stop' if pos.get('phase') == 1 else 'trail'})\n"
                f"Entry:      ${pos['entry']:.2f}\n"
                f"Final exit: {filled} @ ${price:.2f}\n"
                f"TP1 P&L:    ${pos.get('tp_pnl', 0.0):+,.2f}\n"
                f"Final P&L:  ${exit_pnl:+,.2f}\n"
                f"Trade P&L:  ${total_trade_pnl:+,.2f}\n"
                f"Day P&L:    ${self.realized_pnl:+,.2f}\n"
                f"Locked out: {sym in self.locked_out}\n"
                f"Time:       {datetime.now(EASTERN).isoformat()}\n"
            ),
        )

    async def eod_closeout(self) -> None:
        log.info("EOD closeout — day_pnl=$%.2f locked_out=%d", self.realized_pnl, len(self.locked_out))
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
        self.regime_reduced = self._check_regime()
        log.info(
            "equity=$%.2f dry_run=%s lockout=$%.0f hybrid_tp=%.1f%% atr_mult=%.1f "
            "short_mult=%.2f meme_mult=%.2f regime_mult=%.2f regime_reduced=%s blocked=%s",
            self.equity, self.dry_run,
            LOCKOUT_LOSS, HYBRID_TP_PCT, HYBRID_ATR_MULT,
            SHORT_SIZE_MULT, MEME_SIZE_MULT, REGIME_SIZE_MULT,
            self.regime_reduced, sorted(BLOCKED_TICKERS),
        )

        watchlist = self.load_watchlist()
        dropped = [s for s in watchlist if s in BLOCKED_TICKERS]
        if dropped:
            log.info("dropping blocked tickers from watchlist: %s", dropped)
            for s in dropped:
                watchlist.pop(s, None)
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
