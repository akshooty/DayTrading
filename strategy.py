"""Breakout 1-minute bar state machines (pure logic, no I/O).

Two mirror strategies:
  - BreakoutLongState:  uptrend -> pullback -> break above pre-pullback high
  - BreakoutShortState: downtrend -> bounce -> break below pre-bounce low
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union


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
            if self.lh_count >= self.lh_transitions:
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
            if self.hl_count >= self.hl_transitions:
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


BreakoutState = Union[BreakoutLongState, BreakoutShortState]


def position_size(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    min_stop_distance: float = 0.10,
    max_deployment: float = 5000.0,
) -> int:
    risk_dollars = equity * risk_pct
    stop_dist = max(abs(entry - stop), min_stop_distance)
    qty_from_risk = risk_dollars / stop_dist
    qty_from_cap = max_deployment / entry if entry > 0 else 0.0
    return max(int(min(qty_from_risk, qty_from_cap)), 0)
