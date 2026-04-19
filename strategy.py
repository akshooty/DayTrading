"""Breakout 1-minute bar state machines (pure logic, no I/O).

Two mirror strategies:
  - BreakoutLongState:  uptrend -> pullback -> break above pre-pullback high
  - BreakoutShortState: downtrend -> bounce -> break below pre-bounce low
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, Union
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


class State(Enum):
    WATCHING = "watching"
    UPTREND = "uptrend"
    PULLBACK = "pullback"
    DOWNTREND = "downtrend"
    BOUNCE = "bounce"
    ARMED = "armed"
    IN_POSITION = "in_position"
    CLOSED = "closed"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass
class ArmSignal:
    symbol: str
    direction: Direction
    entry: float  # breakout level (above for long, below for short)
    stop: float   # initial stop (pullback low for long, bounce high for short)
    target: Optional[float] = None  # absolute TP price; overrides HYBRID_TP_PCT
    tag: str = ""  # routing tag, e.g. "runner" / "spike" — controls Phase 2 trail
    # Populated by backtest at arm time via compute_signal_score() below.
    # Used by the top-quartile signal-quality filter when enabled.
    score: float = 0.0
    score_components: Optional[dict] = None


@dataclass
class ResetSignal:
    symbol: str
    direction: Direction


Signal = Union[ArmSignal, ResetSignal]


@dataclass
class BreakoutLongState:
    symbol: str
    hh_transitions: int = 3
    lh_transitions: int = 2
    skip_first_n_bars: int = 5

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    intraday_high: float = 0.0
    peak_high: float = 0.0
    pullback_low: float = float("inf")
    lh_count: int = 0

    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if bar.high > self.intraday_high:
            self.intraday_high = bar.high
        if len(self.bars) <= self.skip_first_n_bars:
            return None

        if self.state is State.WATCHING:
            return self._on_watching(bar)
        if self.state is State.UPTREND:
            return self._on_uptrend(bar)
        if self.state is State.PULLBACK:
            return self._on_pullback(bar)
        if self.state is State.ARMED:
            return self._on_armed(bar)
        return None

    def _on_watching(self, bar: Bar) -> Optional[Signal]:
        if self._hh_streak_ok() and bar.high >= self.intraday_high:
            self.state = State.UPTREND
            self.peak_high = bar.high
        return None

    def _on_uptrend(self, bar: Bar) -> Optional[Signal]:
        if bar.high > self.peak_high:
            self.peak_high = bar.high
            return None
        prev = self.bars[-2]
        if bar.high < prev.high:
            self.state = State.PULLBACK
            self.pullback_low = bar.low
            self.lh_count = 1
        return None

    def _on_pullback(self, bar: Bar) -> Optional[Signal]:
        """Pullback: count LH bars, then wait for a reversal (first HH
        bar after the pullback) before arming. ARM fires on the
        reversal bar; the actual entry order waits for price to break
        back above the pre-pullback peak (HOD)."""
        if bar.low < self.pullback_low:
            self.pullback_low = bar.low
        prev = self.bars[-2]
        if bar.high > self.peak_high:
            self.state = State.UPTREND
            self.peak_high = bar.high
            self.lh_count = 0
            return None
        if bar.high < prev.high:
            self.lh_count += 1
            return None
        if bar.high > prev.high and self.lh_count >= self.lh_transitions:
            self.state = State.ARMED
            return ArmSignal(self.symbol, Direction.LONG, self.peak_high, self.pullback_low)
        return None

    def _on_armed(self, bar: Bar) -> Optional[Signal]:
        if bar.low < self.pullback_low:
            self.reset_to_watching()
            return ResetSignal(self.symbol, Direction.LONG)
        return None

    def _hh_streak_ok(self) -> bool:
        needed = self.hh_transitions + 1
        if len(self.bars) < needed:
            return False
        w = self.bars[-needed:]
        return all(w[i].high > w[i - 1].high for i in range(1, needed))

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING
        self.peak_high = 0.0
        self.pullback_low = float("inf")
        self.lh_count = 0

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


@dataclass
class BreakoutShortState:
    symbol: str
    ll_transitions: int = 3
    hl_transitions: int = 2
    skip_first_n_bars: int = 5

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    intraday_low: float = float("inf")
    trough_low: float = float("inf")
    bounce_high: float = 0.0
    hl_count: int = 0

    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if bar.low < self.intraday_low:
            self.intraday_low = bar.low
        if len(self.bars) <= self.skip_first_n_bars:
            return None

        if self.state is State.WATCHING:
            return self._on_watching(bar)
        if self.state is State.DOWNTREND:
            return self._on_downtrend(bar)
        if self.state is State.BOUNCE:
            return self._on_bounce(bar)
        if self.state is State.ARMED:
            return self._on_armed(bar)
        return None

    def _on_watching(self, bar: Bar) -> Optional[Signal]:
        if self._ll_streak_ok() and bar.low <= self.intraday_low:
            self.state = State.DOWNTREND
            self.trough_low = bar.low
        return None

    def _on_downtrend(self, bar: Bar) -> Optional[Signal]:
        if bar.low < self.trough_low:
            self.trough_low = bar.low
            return None
        prev = self.bars[-2]
        if bar.low > prev.low:
            self.state = State.BOUNCE
            self.bounce_high = bar.high
            self.hl_count = 1
        return None

    def _on_bounce(self, bar: Bar) -> Optional[Signal]:
        """Bounce: count HL bars, then wait for reversal (first LL bar
        after the bounce) before arming. ARM fires on the reversal
        bar; entry order waits for price to break below pre-bounce
        trough (LOD)."""
        if bar.high > self.bounce_high:
            self.bounce_high = bar.high
        prev = self.bars[-2]
        if bar.low < self.trough_low:
            self.state = State.DOWNTREND
            self.trough_low = bar.low
            self.hl_count = 0
            return None
        if bar.low > prev.low:
            self.hl_count += 1
            return None
        if bar.low < prev.low and self.hl_count >= self.hl_transitions:
            self.state = State.ARMED
            return ArmSignal(self.symbol, Direction.SHORT, self.trough_low, self.bounce_high)
        return None

    def _on_armed(self, bar: Bar) -> Optional[Signal]:
        if bar.high > self.bounce_high:
            self.reset_to_watching()
            return ResetSignal(self.symbol, Direction.SHORT)
        return None

    def _ll_streak_ok(self) -> bool:
        needed = self.ll_transitions + 1
        if len(self.bars) < needed:
            return False
        w = self.bars[-needed:]
        return all(w[i].low < w[i - 1].low for i in range(1, needed))

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING
        self.trough_low = float("inf")
        self.bounce_high = 0.0
        self.hl_count = 0

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


@dataclass
class ORBState:
    """Opening Range Breakout.

    Track high/low of the first `range_bars` minutes of the session.
    Once range is defined, ARM on first bar whose high > range_high
    (LONG) or low < range_low (SHORT). Stop = opposite side of range.
    """
    symbol: str
    range_bars: int = 10  # first N 1-min bars form the range

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    range_high: float = 0.0
    range_low: float = float("inf")
    range_defined: bool = False

    # direction is set when the range breaks; trader uses it post-fill.
    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)

        if not self.range_defined:
            if bar.high > self.range_high:
                self.range_high = bar.high
            if bar.low < self.range_low:
                self.range_low = bar.low
            if len(self.bars) >= self.range_bars:
                self.range_defined = True
            return None

        if self.state is not State.WATCHING:
            return None

        # Range defined; watch for breakout. Long takes precedence if a
        # single bar straddles both boundaries (rare).
        if bar.high > self.range_high:
            self.state = State.ARMED
            self.direction = Direction.LONG
            return ArmSignal(self.symbol, Direction.LONG, self.range_high, self.range_low)
        if bar.low < self.range_low:
            self.state = State.ARMED
            self.direction = Direction.SHORT
            return ArmSignal(self.symbol, Direction.SHORT, self.range_low, self.range_high)
        return None

    def reset_to_watching(self) -> None:
        # ORB is one-shot per day: once the range breaks and we miss
        # the entry (cap full, ticker busy, chop window), the setup is
        # gone. Terminate to prevent re-arming every subsequent bar.
        self.state = State.CLOSED

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


def _atr_of(bars: list[Bar], period: int) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    recent = bars[-(period + 1):]
    for i in range(1, len(recent)):
        b, prev = recent[i], recent[i - 1]
        tr = max(b.high - b.low, abs(b.high - prev.close), abs(b.low - prev.close))
        trs.append(tr)
    return sum(trs) / period


def _ema_update(prev: Optional[float], price: float, seed: list[float], period: int) -> Optional[float]:
    """Incremental EMA. Returns new EMA, or None if still seeding."""
    if prev is None:
        seed.append(price)
        if len(seed) >= period:
            return sum(seed[-period:]) / period
        return None
    k = 2.0 / (period + 1)
    return price * k + prev * (1 - k)


@dataclass
class RunnerLong:
    """Momentum continuation on strong intraday movers (LONG).

    Setup:
      - Intraday high >= +move_threshold above session open ("runner")
      - Pullback from HOD to 1-min EMA(9) and consolidation
      - >= cons_bars_required consecutive tight bars near the 9 EMA
    Entry: break above consolidation high.
    Stop:  consolidation low.
    TP1:   half qty at entry + rr_target * risk (1:1.6 R by default).
    Trail: after TP1, stop = midpoint(current extreme, entry), ratcheting.
    """
    symbol: str
    move_threshold: float = 0.05
    ema_period: int = 9
    cons_bars_required: int = 5
    cons_proximity_pct: float = 0.01
    cons_tight_atr_mult: float = 0.6
    rr_target: float = 1.6
    skip_first_n_bars: int = 5

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    session_open: float = 0.0
    intraday_high: float = 0.0
    qualified: bool = False
    ema9: Optional[float] = None
    ema_seed: list[float] = field(default_factory=list)
    cons_count: int = 0
    cons_high: float = 0.0
    cons_low: float = float("inf")

    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if self.session_open == 0.0:
            self.session_open = bar.open
        if bar.high > self.intraday_high:
            self.intraday_high = bar.high

        self.ema9 = _ema_update(self.ema9, bar.close, self.ema_seed, self.ema_period)
        if len(self.bars) <= self.skip_first_n_bars or self.ema9 is None:
            return None

        if not self.qualified and self.session_open > 0:
            if (self.intraday_high - self.session_open) / self.session_open >= self.move_threshold:
                self.qualified = True
        if not self.qualified:
            return None

        atr = _atr_of(self.bars, 14)
        if atr <= 0:
            return None
        near_ema = abs(bar.close - self.ema9) / self.ema9 <= self.cons_proximity_pct
        tight = (bar.high - bar.low) <= atr * self.cons_tight_atr_mult

        if self.state is State.WATCHING:
            if near_ema and bar.close < self.intraday_high:
                self.state = State.PULLBACK
                self.cons_count = 1
                self.cons_high = bar.high
                self.cons_low = bar.low
            return None

        if self.state is State.PULLBACK:
            if bar.close > self.cons_high and self.cons_count >= self.cons_bars_required:
                entry = self.cons_high
                stop = self.cons_low
                risk = entry - stop
                if risk <= 0:
                    self.reset_to_watching()
                    return None
                target = entry + self.rr_target * risk
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.LONG,
                    entry=entry, stop=stop,
                    target=target, tag="runner",
                )
            if near_ema and tight:
                self.cons_count += 1
                if bar.high > self.cons_high:
                    self.cons_high = bar.high
                if bar.low < self.cons_low:
                    self.cons_low = bar.low
            elif bar.close < self.cons_low * 0.995:
                self.reset_to_watching()
            return None

        if self.state is State.ARMED:
            if bar.low < self.cons_low:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.LONG)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING
        self.cons_count = 0
        self.cons_high = 0.0
        self.cons_low = float("inf")

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED



@dataclass
class RunnerShort:
    """Mirror of RunnerLong: momentum continuation on strong DOWN movers.

    Setup:
      - Intraday low >= move_threshold BELOW session open ("faller")
      - Bounce from LOD up to 1-min EMA(9) and consolidation
      - >= cons_bars_required consecutive tight bars near the 9 EMA
    Entry: break BELOW consolidation low.
    Stop:  consolidation high.
    TP1:   half qty at entry - rr_target * risk (1:1.6 R).
    Trail: after TP1, stop = midpoint(current extreme-low, entry), ratcheting.
    """
    symbol: str
    move_threshold: float = 0.05
    ema_period: int = 9
    cons_bars_required: int = 5
    cons_proximity_pct: float = 0.01
    cons_tight_atr_mult: float = 0.6
    rr_target: float = 1.6
    skip_first_n_bars: int = 5

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    session_open: float = 0.0
    intraday_low: float = float("inf")
    qualified: bool = False
    ema9: Optional[float] = None
    ema_seed: list[float] = field(default_factory=list)
    cons_count: int = 0
    cons_high: float = 0.0
    cons_low: float = float("inf")

    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if self.session_open == 0.0:
            self.session_open = bar.open
        if bar.low < self.intraday_low:
            self.intraday_low = bar.low

        self.ema9 = _ema_update(self.ema9, bar.close, self.ema_seed, self.ema_period)
        if len(self.bars) <= self.skip_first_n_bars or self.ema9 is None:
            return None

        if not self.qualified and self.session_open > 0:
            if (self.session_open - self.intraday_low) / self.session_open >= self.move_threshold:
                self.qualified = True
        if not self.qualified:
            return None

        atr = _atr_of(self.bars, 14)
        if atr <= 0:
            return None
        near_ema = abs(bar.close - self.ema9) / self.ema9 <= self.cons_proximity_pct
        tight = (bar.high - bar.low) <= atr * self.cons_tight_atr_mult

        if self.state is State.WATCHING:
            if near_ema and bar.close > self.intraday_low:
                self.state = State.BOUNCE
                self.cons_count = 1
                self.cons_high = bar.high
                self.cons_low = bar.low
            return None

        if self.state is State.BOUNCE:
            if bar.close < self.cons_low and self.cons_count >= self.cons_bars_required:
                entry = self.cons_low
                stop = self.cons_high
                risk = stop - entry
                if risk <= 0:
                    self.reset_to_watching()
                    return None
                target = entry - self.rr_target * risk
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.SHORT,
                    entry=entry, stop=stop,
                    target=target, tag="runner",
                )
            if near_ema and tight:
                self.cons_count += 1
                if bar.high > self.cons_high:
                    self.cons_high = bar.high
                if bar.low < self.cons_low:
                    self.cons_low = bar.low
            elif bar.close > self.cons_high * 1.005:
                self.reset_to_watching()
            return None

        if self.state is State.ARMED:
            if bar.high > self.cons_high:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.SHORT)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING
        self.cons_count = 0
        self.cons_high = 0.0
        self.cons_low = float("inf")

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED










@dataclass
class VWAPSnapbackLong:
    """Mean-reversion scalp LONG: fade stretched downside moves back to VWAP.

    Setup: price extended >= entry_sigma_threshold volume-weighted std-devs
    BELOW session VWAP, followed by a bullish reversal candle (close > open)
    that still closes below VWAP.
    Entry: bar.close (market on next bar's open).
    Stop:  bar.low minus a small buffer.
    Target: VWAP at arm time.
    """
    symbol: str
    entry_sigma_threshold: float = 1.75
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    vwap_num: float = 0.0
    vwap_den: float = 0.0
    vwap_sq_num: float = 0.0
    pending_entry: float = 0.0
    pending_stop: float = 0.0

    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        typ = (bar.high + bar.low + bar.close) / 3.0
        vol = max(bar.volume, 0.0)
        self.vwap_num += typ * vol
        self.vwap_den += vol
        self.vwap_sq_num += typ * typ * vol
        if self.vwap_den <= 0 or len(self.bars) <= self.skip_first_n_bars:
            return None
        vwap = self.vwap_num / self.vwap_den
        variance = self.vwap_sq_num / self.vwap_den - vwap * vwap
        if variance <= 0:
            return None
        sigma = variance ** 0.5

        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None

        if self.state is State.WATCHING:
            if (
                bar.low <= vwap - self.entry_sigma_threshold * sigma
                and bar.close > bar.open
                and bar.close < vwap
            ):
                entry = bar.close
                stop = bar.low * (1 - self.stop_buffer_pct)
                target = vwap
                if target - entry <= 0 or entry - stop <= 0:
                    return None
                self.pending_entry = entry
                self.pending_stop = stop
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.LONG,
                    entry=entry, stop=stop,
                    target=target, tag="vwap_snap",
                )
            return None

        if self.state is State.ARMED:
            if bar.low < self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.LONG)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


@dataclass
class VWAPSnapbackShort:
    """Mirror of VWAPSnapbackLong: fade stretched UP moves back to VWAP."""
    symbol: str
    entry_sigma_threshold: float = 1.75
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    vwap_num: float = 0.0
    vwap_den: float = 0.0
    vwap_sq_num: float = 0.0
    pending_entry: float = 0.0
    pending_stop: float = 0.0

    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        typ = (bar.high + bar.low + bar.close) / 3.0
        vol = max(bar.volume, 0.0)
        self.vwap_num += typ * vol
        self.vwap_den += vol
        self.vwap_sq_num += typ * typ * vol
        if self.vwap_den <= 0 or len(self.bars) <= self.skip_first_n_bars:
            return None
        vwap = self.vwap_num / self.vwap_den
        variance = self.vwap_sq_num / self.vwap_den - vwap * vwap
        if variance <= 0:
            return None
        sigma = variance ** 0.5

        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None

        if self.state is State.WATCHING:
            if (
                bar.high >= vwap + self.entry_sigma_threshold * sigma
                and bar.close < bar.open
                and bar.close > vwap
            ):
                entry = bar.close
                stop = bar.high * (1 + self.stop_buffer_pct)
                target = vwap
                if entry - target <= 0 or stop - entry <= 0:
                    return None
                self.pending_entry = entry
                self.pending_stop = stop
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.SHORT,
                    entry=entry, stop=stop,
                    target=target, tag="vwap_snap",
                )
            return None

        if self.state is State.ARMED:
            if bar.high > self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.SHORT)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


@dataclass
class CatalystLong:
    """Event-proxy LONG: volume spike + directional break of recent range.

    We don't have a macro-event calendar, so we use ticker-level volume +
    SPY-gap regime as a microstructure proxy for "something is happening".
      - Current bar volume >= vol_spike_mult * rolling avg of last vol_lookback bars
      - Current bar close > max(prior range_lookback bars' highs)
      - Bullish bar (close > open)
    On SPY-gap days (runtime sets sky_gap_active=True), threshold is
    relaxed by relax_factor.
    Entry: bar.close (market next bar).
    Stop:  min(last range_lookback bars' lows) minus buffer.
    Target: entry + rr_target * risk.
    """
    symbol: str
    vol_spike_mult: float = 4.0
    vol_lookback: int = 30
    range_lookback: int = 10
    rr_target: float = 2.0
    stop_buffer_pct: float = 0.002
    relax_factor: float = 0.75
    skip_first_n_bars: int = 30
    entry_end_et: time = time(15, 0)  # no new entries after 15:00 ET (EOD cutoff)

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    sky_gap_active: bool = False

    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if len(self.bars) <= max(self.skip_first_n_bars, self.vol_lookback + 1):
            return None

        # EOD cutoff: still let ARMED stops fire, but no new arms.
        et = bar.timestamp.astimezone(_EASTERN).time()
        eod_block = et >= self.entry_end_et

        vol_window = self.bars[-(self.vol_lookback + 1):-1]
        avg_vol = sum(b.volume for b in vol_window) / max(len(vol_window), 1)
        if avg_vol <= 0:
            if self.state is State.ARMED and bar.low < self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.LONG)
            return None
        threshold = self.vol_spike_mult * (
            self.relax_factor if self.sky_gap_active else 1.0
        )
        spike = bar.volume >= threshold * avg_vol

        if self.state is State.WATCHING:
            if eod_block or not spike:
                return None
            range_window = self.bars[-(self.range_lookback + 1):-1]
            prior_high = max(b.high for b in range_window)
            prior_low = min(b.low for b in range_window)
            if bar.close > prior_high and bar.close > bar.open:
                entry = bar.close
                stop = prior_low * (1 - self.stop_buffer_pct)
                risk = entry - stop
                if risk <= 0:
                    return None
                target = entry + self.rr_target * risk
                self.pending_entry = entry
                self.pending_stop = stop
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.LONG,
                    entry=entry, stop=stop,
                    target=target, tag="catalyst",
                )
            return None

        if self.state is State.ARMED:
            if bar.low < self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.LONG)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED


@dataclass
class CatalystShort:
    """Mirror of CatalystLong: volume spike + break DOWN of recent range."""
    symbol: str
    vol_spike_mult: float = 4.0
    vol_lookback: int = 30
    range_lookback: int = 10
    rr_target: float = 2.0
    stop_buffer_pct: float = 0.002
    relax_factor: float = 0.75
    skip_first_n_bars: int = 30
    entry_end_et: time = time(15, 0)  # EOD cutoff

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    sky_gap_active: bool = False

    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if len(self.bars) <= max(self.skip_first_n_bars, self.vol_lookback + 1):
            return None

        et = bar.timestamp.astimezone(_EASTERN).time()
        eod_block = et >= self.entry_end_et

        vol_window = self.bars[-(self.vol_lookback + 1):-1]
        avg_vol = sum(b.volume for b in vol_window) / max(len(vol_window), 1)
        if avg_vol <= 0:
            if self.state is State.ARMED and bar.high > self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.SHORT)
            return None
        threshold = self.vol_spike_mult * (
            self.relax_factor if self.sky_gap_active else 1.0
        )
        spike = bar.volume >= threshold * avg_vol

        if self.state is State.WATCHING:
            if eod_block or not spike:
                return None
            range_window = self.bars[-(self.range_lookback + 1):-1]
            prior_high = max(b.high for b in range_window)
            prior_low = min(b.low for b in range_window)
            if bar.close < prior_low and bar.close < bar.open:
                entry = bar.close
                stop = prior_high * (1 + self.stop_buffer_pct)
                risk = stop - entry
                if risk <= 0:
                    return None
                target = entry - self.rr_target * risk
                self.pending_entry = entry
                self.pending_stop = stop
                self.state = State.ARMED
                return ArmSignal(
                    self.symbol, Direction.SHORT,
                    entry=entry, stop=stop,
                    target=target, tag="catalyst",
                )
            return None

        if self.state is State.ARMED:
            if bar.high > self.pending_stop:
                self.reset_to_watching()
                return ResetSignal(self.symbol, Direction.SHORT)
            return None
        return None

    def reset_to_watching(self) -> None:
        self.state = State.WATCHING

    def on_entry_filled(self) -> None:
        self.state = State.IN_POSITION

    def on_exit_filled(self) -> None:
        self.state = State.CLOSED





BreakoutState = Union[
    BreakoutLongState, BreakoutShortState, ORBState,
    RunnerLong, RunnerShort,
    VWAPSnapbackLong, VWAPSnapbackShort,
    CatalystLong, CatalystShort,
]


def min_stop_dist(entry: float, floor: float = 0.01, pct: float = 0.01) -> float:
    """Proportional minimum stop distance: max(1 cent, 1% of entry).

    A flat $0.10 floor works for $10+ stocks but is absurdly wide for
    sub-$1 stocks (22% of a $0.46 SAFX) and absurdly tight for $100+
    stocks (0.1% of $100 MSTR). This scales with price.
    """
    return max(floor, pct * entry)


def position_size(
    entry: float,
    stop: float,
    min_stop_distance: float | None = None,
    max_deployment: float = 5000.0,
    max_risk: float = 200.0,
) -> int:
    """Size a position so that IF the initial stop hits, loss == max_risk.

    Also bounded by max_deployment ($ notional) to keep low-priced names
    from blowing up share counts.
    """
    if min_stop_distance is None:
        min_stop_distance = min_stop_dist(entry)
    stop_dist = max(abs(entry - stop), min_stop_distance)
    qty_from_risk = max_risk / stop_dist
    qty_from_cap = max_deployment / entry if entry > 0 else 0.0
    return max(int(min(qty_from_risk, qty_from_cap)), 0)


# ---------------------------------------------------------------------------
# Signal-quality scoring (change #1: top-quartile filter)
# ---------------------------------------------------------------------------
#
# Composite score combining features computable from the strategy's own bar
# history at arm time. NO look-ahead: only bars up to and including the arm
# bar are used. Higher = higher conviction.
#
# Features:
#   relvol      — arm-bar volume / mean volume over the prior 20 bars.
#                 Strong relvol = institutional interest, cleaner breakout.
#   push        — distance price extended beyond the trigger level on the arm
#                 bar, normalized by ATR(14). Bigger push = stronger momentum.
#   atr_expand  — recent ATR(14) vs. ATR(14) from 15 bars earlier.
#                 >1 = volatility expanding (fresh move), <1 = compressing.
#   intraday    — arm-bar move vs. session open, normalized by session ATR.
#                 Proxy for "catalyst strength / price action" — big daily
#                 movers score high.
#   gap         — session open vs. yesterday's close. Not computable from
#                 intraday bars alone (prior-day close not in state). When a
#                 prior-close reference is unavailable, returns 0 and is
#                 noted in score_components.
#
# Float isn't computable from bar data; a proper implementation needs an
# external float/shares-outstanding data source (left as TODO).
#
# The backtest's two-pass mode (COLLECT_SIGNALS_ONLY then MIN_SIGNAL_SCORE)
# computes the 75th-percentile threshold from pass 1 and applies it in pass 2.
# ---------------------------------------------------------------------------

def _tr(bar, prev_close):
    return max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))


def _atr(bars, period):
    if len(bars) < period + 1:
        return 0.0
    recent = bars[-(period + 1):]
    trs = [_tr(recent[i], recent[i - 1].close) for i in range(1, len(recent))]
    return sum(trs) / period


def compute_signal_score(
    bars: list,
    direction: "Direction",
    entry: float,
    stop: float,
    prior_day_close: Optional[float] = None,
) -> tuple[float, dict]:
    """Return (composite_score, components_dict).

    `bars` should be the strategy's bar history including the arm bar.
    """
    comp: dict = {"relvol": 0.0, "push": 0.0, "atr_expand": 0.0,
                  "intraday": 0.0, "gap": 0.0, "float": None}
    if not bars:
        return 0.0, comp
    arm_bar = bars[-1]

    # relvol: arm-bar volume vs prior-20 mean
    prior = bars[-21:-1] if len(bars) >= 21 else bars[:-1]
    if prior:
        mean_vol = sum(b.volume for b in prior) / len(prior)
        if mean_vol > 0:
            comp["relvol"] = arm_bar.volume / mean_vol

    # push: how far past trigger did this bar go, normalized by ATR(14)
    atr14 = _atr(bars, 14)
    if atr14 > 0:
        if direction.value == "long":
            push_abs = max(0.0, arm_bar.high - entry)
        else:
            push_abs = max(0.0, entry - arm_bar.low)
        comp["push"] = push_abs / atr14

    # atr expansion: ATR(14) now vs ATR(14) ending 15 bars ago
    if len(bars) >= 30:
        atr_prev = _atr(bars[:-15], 14)
        if atr_prev > 0:
            comp["atr_expand"] = atr14 / atr_prev

    # intraday move: arm bar close vs session open, normalized by atr14
    session_open = bars[0].open
    if atr14 > 0 and session_open > 0:
        move = (arm_bar.close - session_open) / atr14
        # Long wants positive move, short wants negative.
        comp["intraday"] = move if direction.value == "long" else -move

    # gap: session open vs prior-day close (only if caller provided)
    if prior_day_close and prior_day_close > 0 and session_open > 0:
        gap = (session_open - prior_day_close) / prior_day_close
        comp["gap"] = gap if direction.value == "long" else -gap

    # Composite: sum of standardized-ish components. Weights chosen to put
    # each component on a comparable scale (relvol and intraday tend to be
    # O(1-10); push O(0-3); atr_expand O(0.5-2); gap O(0.0-0.15)).
    score = (
        1.0 * comp["relvol"]
        + 2.0 * comp["push"]
        + 1.5 * comp["atr_expand"]
        + 1.0 * comp["intraday"]
        + 10.0 * comp["gap"]
    )
    return score, comp
