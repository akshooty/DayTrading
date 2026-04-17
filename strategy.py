"""Breakout-long 1-minute bar state machine (pure logic, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union


class State(Enum):
    WATCHING = "watching"
    UPTREND = "uptrend"
    PULLBACK = "pullback"
    ARMED = "armed"
    IN_POSITION = "in_position"
    CLOSED = "closed"


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
    entry: float
    stop: float


@dataclass
class ResetSignal:
    symbol: str


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
                return ArmSignal(self.symbol, self.peak_high, self.pullback_low)
        return None

    def _on_armed(self, bar: Bar) -> Optional[Signal]:
        if bar.low < self.pullback_low:
            self.reset_to_watching()
            return ResetSignal(self.symbol)
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


def position_size(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    min_stop_distance: float = 0.10,
) -> int:
    risk_dollars = equity * risk_pct
    stop_dist = max(entry - stop, min_stop_distance)
    return max(int(risk_dollars / stop_dist), 0)
