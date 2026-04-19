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
    cons_bars_required: int = 3
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
    cons_bars_required: int = 3
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





def _l2l_levels(bars: list[Bar], or_bars: int, current_price: float) -> list[float]:
    """Compute key levels dynamically: session open, intraday H/L,
    opening-range H/L (first or_bars bars), and round-dollar numbers
    within +/- 5% of current price. Returned sorted ascending.
    """
    if not bars:
        return []
    levels = set()
    levels.add(round(bars[0].open, 4))
    levels.add(round(max(b.high for b in bars), 4))
    levels.add(round(min(b.low for b in bars), 4))
    if len(bars) >= or_bars:
        or_window = bars[:or_bars]
        levels.add(round(max(b.high for b in or_window), 4))
        levels.add(round(min(b.low for b in or_window), 4))
    band_lo = current_price * 0.95
    band_hi = current_price * 1.05
    if current_price >= 100:
        step = 1.0
    elif current_price >= 10:
        step = 0.5
    elif current_price >= 1:
        step = 0.10
    else:
        step = 0.05
    n = int(band_lo / step)
    while n * step <= band_hi:
        if n * step >= band_lo:
            levels.add(round(n * step, 4))
        n += 1
    return sorted(levels)


@dataclass
class LevelToLevelLong:
    """Mid-day level-to-level scalp LONG. Active only during chop window
    (default 11:30-14:00 ET). Levels computed dynamically each bar:
    session open, opening-range H/L, intraday H/L, round-dollar numbers
    within +/- 5% of current price.

    Setup: bar.low touches/penetrates a level from above AND bar.close > level
    (rejection — buyers defended).
    Entry: bar.close (market on next bar's open).
    Stop:  rejection bar's low minus small buffer.
    Target: next level above current price.
    """
    symbol: str
    or_bars: int = 30
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    touch_tolerance_pct: float = 0.003
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if len(self.bars) <= self.skip_first_n_bars:
            return None
        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None
        levels = _l2l_levels(self.bars, self.or_bars, bar.close)
        if not levels:
            return None

        if self.state is State.WATCHING:
            for lvl in levels:
                tol = lvl * self.touch_tolerance_pct
                if (
                    bar.low <= lvl + tol
                    and bar.close > lvl
                    and bar.close < lvl * 1.01
                ):
                    higher = [l for l in levels if l > bar.close * 1.002]
                    if not higher:
                        continue
                    target = higher[0]
                    entry = bar.close
                    stop = bar.low * (1 - self.stop_buffer_pct)
                    if target - entry <= 0 or entry - stop <= 0:
                        continue
                    self.pending_entry = entry
                    self.pending_stop = stop
                    self.state = State.ARMED
                    return ArmSignal(
                        self.symbol, Direction.LONG,
                        entry=entry, stop=stop,
                        target=target, tag="l2l",
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
class LevelToLevelShort:
    """Mirror of LevelToLevelLong."""
    symbol: str
    or_bars: int = 30
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    touch_tolerance_pct: float = 0.003
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self.bars.append(bar)
        if len(self.bars) <= self.skip_first_n_bars:
            return None
        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None
        levels = _l2l_levels(self.bars, self.or_bars, bar.close)
        if not levels:
            return None

        if self.state is State.WATCHING:
            for lvl in levels:
                tol = lvl * self.touch_tolerance_pct
                if (
                    bar.high >= lvl - tol
                    and bar.close < lvl
                    and bar.close > lvl * 0.99
                ):
                    lower = sorted((l for l in levels if l < bar.close * 0.998), reverse=True)
                    if not lower:
                        continue
                    target = lower[0]
                    entry = bar.close
                    stop = bar.high * (1 + self.stop_buffer_pct)
                    if entry - target <= 0 or stop - entry <= 0:
                        continue
                    self.pending_entry = entry
                    self.pending_stop = stop
                    self.state = State.ARMED
                    return ArmSignal(
                        self.symbol, Direction.SHORT,
                        entry=entry, stop=stop,
                        target=target, tag="l2l",
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
class SectorRotationLong:
    """Mid-day relative-strength continuation LONG. Active in chop window
    (default 11:30-14:00 ET). Triggers when the ticker prints a new HOD
    while the broad index is flat (|SPY intraday move| < flat_pct).

    SPY intraday move is read from the `index_move_pct` attribute, which
    the runtime updates each bar. If the runtime never updates it, the
    SPY-flat gate is skipped (falls back to 'new HOD in chop' only).

    Setup: bar.high > prior intraday HOD AND |index_move_pct| < flat_pct
    Entry: bar.close (market next bar)
    Stop:  bar.low - small buffer
    Target: entry + rr_target * risk
    """
    symbol: str
    flat_pct: float = 0.003          # SPY |move| < 0.3% counts as flat
    rr_target: float = 2.0
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    intraday_high: float = 0.0
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    index_move_pct: Optional[float] = None  # set externally per bar

    direction: Direction = Direction.LONG

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        prior_hod = self.intraday_high
        self.bars.append(bar)
        if bar.high > self.intraday_high:
            self.intraday_high = bar.high
        if len(self.bars) <= self.skip_first_n_bars:
            return None
        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None

        if self.state is State.WATCHING:
            if bar.high <= prior_hod:
                return None
            # SPY-flat gate (only enforced when index_move_pct is provided).
            if self.index_move_pct is not None and abs(self.index_move_pct) > self.flat_pct:
                return None
            entry = bar.close
            stop = bar.low * (1 - self.stop_buffer_pct)
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
                target=target, tag="sector",
            )

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
class SectorRotationShort:
    """Mirror of SectorRotationLong: new intraday LOW with SPY flat = fade short."""
    symbol: str
    flat_pct: float = 0.003
    rr_target: float = 2.0
    active_start_et: time = time(11, 30)
    active_end_et: time = time(14, 0)
    stop_buffer_pct: float = 0.002
    skip_first_n_bars: int = 30

    state: State = State.WATCHING
    bars: list[Bar] = field(default_factory=list)
    intraday_low: float = float("inf")
    pending_entry: float = 0.0
    pending_stop: float = 0.0
    index_move_pct: Optional[float] = None

    direction: Direction = Direction.SHORT

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        prior_lod = self.intraday_low
        self.bars.append(bar)
        if bar.low < self.intraday_low:
            self.intraday_low = bar.low
        if len(self.bars) <= self.skip_first_n_bars:
            return None
        et = bar.timestamp.astimezone(_EASTERN).time()
        if not (self.active_start_et <= et < self.active_end_et):
            return None

        if self.state is State.WATCHING:
            if bar.low >= prior_lod:
                return None
            if self.index_move_pct is not None and abs(self.index_move_pct) > self.flat_pct:
                return None
            entry = bar.close
            stop = bar.high * (1 + self.stop_buffer_pct)
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
                target=target, tag="sector",
            )

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
    vol_spike_mult: float = 3.0
    vol_lookback: int = 30
    range_lookback: int = 10
    rr_target: float = 2.0
    stop_buffer_pct: float = 0.002
    relax_factor: float = 0.75
    skip_first_n_bars: int = 30

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
            if not spike:
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
    vol_spike_mult: float = 3.0
    vol_lookback: int = 30
    range_lookback: int = 10
    rr_target: float = 2.0
    stop_buffer_pct: float = 0.002
    relax_factor: float = 0.75
    skip_first_n_bars: int = 30

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
            if not spike:
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
    LevelToLevelLong, LevelToLevelShort,
    SectorRotationLong, SectorRotationShort,
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
