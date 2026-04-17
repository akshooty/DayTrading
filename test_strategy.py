"""Synthetic-bar unit tests for the breakout state machines."""

from datetime import datetime, timedelta

from strategy import (
    ArmSignal,
    Bar,
    BreakoutLongState,
    BreakoutShortState,
    Direction,
    ResetSignal,
    State,
    position_size,
)


def mk(high: float, low: float, ts_offset: int = 0) -> Bar:
    t = datetime(2026, 4, 17, 8, 30) + timedelta(minutes=ts_offset)
    return Bar(open=low, high=high, low=low, close=(high + low) / 2, volume=1000, timestamp=t)


def test_long_happy_path():
    st = BreakoutLongState("TEST")
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        st.on_bar(mk(h, l, i))
    assert st.state is State.WATCHING

    for i, (h, l) in enumerate([(105, 103), (106, 104), (107, 105), (108, 106)], start=5):
        st.on_bar(mk(h, l, i))
    assert st.state is State.UPTREND
    assert st.peak_high == 108

    st.on_bar(mk(107.5, 105.5, 9))
    assert st.state is State.PULLBACK
    assert st.lh_count == 1

    sig = st.on_bar(mk(107, 105, 10))
    assert isinstance(sig, ArmSignal)
    assert sig.direction is Direction.LONG
    assert sig.entry == 108 and sig.stop == 105
    assert st.state is State.ARMED

    st.on_entry_filled()
    assert st.state is State.IN_POSITION
    st.on_exit_filled()
    assert st.state is State.CLOSED
    print("✓ long happy path")


def test_long_reset_on_breach():
    st = BreakoutLongState("TEST")
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(105, 103), (106, 104), (107, 105), (108, 106)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(107.5, 105.5, 9))
    assert isinstance(st.on_bar(mk(107, 105, 10)), ArmSignal)

    reset = st.on_bar(mk(106, 104.5, 11))
    assert isinstance(reset, ResetSignal)
    assert reset.direction is Direction.LONG
    assert st.state is State.WATCHING
    print("✓ long reset on pullback breach")


def test_long_aborts_on_new_high():
    st = BreakoutLongState("TEST")
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(105, 103), (106, 104), (107, 105), (108, 106)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(107.5, 105.5, 9))
    assert st.state is State.PULLBACK

    sig = st.on_bar(mk(109, 107, 10))
    assert sig is None
    assert st.state is State.UPTREND
    assert st.peak_high == 109
    print("✓ long aborts on new high")


def test_short_happy_path():
    st = BreakoutShortState("TEST")
    # First 5 bars: skipped
    for i, (h, l) in enumerate([(100, 99), (100, 98), (99, 97), (98, 96), (97, 95)]):
        st.on_bar(mk(h, l, i))
    assert st.state is State.WATCHING

    # Bars 6-9: strictly-falling lows → downtrend confirmed
    for i, (h, l) in enumerate([(96, 94), (95, 93), (94, 92), (93, 91)], start=5):
        st.on_bar(mk(h, l, i))
    assert st.state is State.DOWNTREND
    assert st.trough_low == 91

    # First HL starts the bounce
    st.on_bar(mk(94, 92, 9))
    assert st.state is State.BOUNCE
    assert st.bounce_high == 94
    assert st.hl_count == 1

    # Second HL confirms bounce → ARM
    sig = st.on_bar(mk(95, 93, 10))
    assert isinstance(sig, ArmSignal)
    assert sig.direction is Direction.SHORT
    assert sig.entry == 91  # breakdown level = pre-bounce trough
    assert sig.stop == 95   # reversal high during bounce
    assert st.state is State.ARMED

    st.on_entry_filled()
    assert st.state is State.IN_POSITION
    st.on_exit_filled()
    assert st.state is State.CLOSED
    print("✓ short happy path")


def test_short_reset_on_breach():
    st = BreakoutShortState("TEST")
    for i, (h, l) in enumerate([(100, 99), (100, 98), (99, 97), (98, 96), (97, 95)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(96, 94), (95, 93), (94, 92), (93, 91)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(94, 92, 9))
    sig = st.on_bar(mk(95, 93, 10))
    assert isinstance(sig, ArmSignal)

    # Bar breaches bounce_high (95) → reset
    reset = st.on_bar(mk(95.5, 93.5, 11))
    assert isinstance(reset, ResetSignal)
    assert reset.direction is Direction.SHORT
    assert st.state is State.WATCHING
    print("✓ short reset on bounce breach")


def test_short_aborts_on_new_low():
    st = BreakoutShortState("TEST")
    for i, (h, l) in enumerate([(100, 99), (100, 98), (99, 97), (98, 96), (97, 95)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(96, 94), (95, 93), (94, 92), (93, 91)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(94, 92, 9))
    assert st.state is State.BOUNCE

    # New lower low → back to DOWNTREND
    sig = st.on_bar(mk(93, 90, 10))
    assert sig is None
    assert st.state is State.DOWNTREND
    assert st.trough_low == 90
    print("✓ short aborts on new low")


def test_position_size():
    # $5000 max deployment cap clamps large risk-based positions.
    # risk 500 / stop 0.50 = 1000 shares, but 5000/$10 = 500 share cap wins.
    assert position_size(100_000, 0.005, 10.00, 9.50) == 500
    assert position_size(100_000, 0.005, 9.50, 10.00) == 526  # short: 5000/9.50
    # Tiny stop → risk formula would give 5000 shares; cap at 500.
    assert position_size(100_000, 0.005, 10.00, 10.00) == 500
    # $2 stock with $1 stop: risk 500 / 1 = 500 shares ($1000) beats 5000/$2=2500 cap.
    assert position_size(100_000, 0.005, 2.00, 1.00) == 500
    assert position_size(0, 0.005, 10.00, 9.50) == 0
    # max_deployment override
    assert position_size(100_000, 0.005, 10.00, 9.50, max_deployment=10_000) == 1000
    print("✓ position sizing (long + short, $5k cap)")


if __name__ == "__main__":
    test_long_happy_path()
    test_long_reset_on_breach()
    test_long_aborts_on_new_high()
    test_short_happy_path()
    test_short_reset_on_breach()
    test_short_aborts_on_new_low()
    test_position_size()
    print("\nall tests passed")
