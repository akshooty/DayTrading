"""Synthetic-bar unit test for the breakout-long state machine."""

from datetime import datetime, timedelta

from strategy import ArmSignal, Bar, BreakoutLongState, ResetSignal, State, position_size


def mk(high: float, low: float, ts_offset: int = 0) -> Bar:
    t = datetime(2026, 4, 17, 8, 30) + timedelta(minutes=ts_offset)
    return Bar(open=low, high=high, low=low, close=(high + low) / 2, volume=1000, timestamp=t)


def test_happy_path_arm_then_fill_then_exit():
    st = BreakoutLongState("TEST")

    # First 5 bars: skipped (skip_first_n_bars = 5)
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        assert st.on_bar(mk(h, l, i)) is None
    assert st.state is State.WATCHING

    # Bars 6-9: strictly-rising highs → uptrend confirmed
    # Need hh_transitions + 1 = 4 bars in strictly-rising sequence
    assert st.on_bar(mk(105, 103, 5)) is None
    assert st.on_bar(mk(106, 104, 6)) is None
    assert st.on_bar(mk(107, 105, 7)) is None
    assert st.on_bar(mk(108, 106, 8)) is None  # HH streak satisfied; new intraday high
    assert st.state is State.UPTREND
    assert st.peak_high == 108

    # Lower-high bar starts the pullback
    assert st.on_bar(mk(107.5, 105.5, 9)) is None
    assert st.state is State.PULLBACK
    assert st.pullback_low == 105.5
    assert st.lh_count == 1

    # Second LH bar confirms pullback → ARM signal
    sig = st.on_bar(mk(107, 105, 10))
    assert isinstance(sig, ArmSignal), f"expected ArmSignal, got {sig}"
    assert sig.entry == 108
    assert sig.stop == 105  # updated pullback low
    assert st.state is State.ARMED

    # Entry fill → IN_POSITION
    st.on_entry_filled()
    assert st.state is State.IN_POSITION

    # Exit fill → CLOSED
    st.on_exit_filled()
    assert st.state is State.CLOSED
    print("✓ happy path")


def test_reset_on_pullback_breach_while_armed():
    st = BreakoutLongState("TEST")
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(105, 103), (106, 104), (107, 105), (108, 106)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(107.5, 105.5, 9))
    sig = st.on_bar(mk(107, 105, 10))
    assert isinstance(sig, ArmSignal)

    # Now a bar that breaches the pullback low (105) → reset
    reset_sig = st.on_bar(mk(106, 104.5, 11))
    assert isinstance(reset_sig, ResetSignal)
    assert st.state is State.WATCHING
    print("✓ pullback-breach reset")


def test_pullback_aborts_if_new_high():
    st = BreakoutLongState("TEST")
    for i, (h, l) in enumerate([(100, 99), (101, 99), (102, 100), (103, 101), (104, 102)]):
        st.on_bar(mk(h, l, i))
    for i, (h, l) in enumerate([(105, 103), (106, 104), (107, 105), (108, 106)], start=5):
        st.on_bar(mk(h, l, i))
    st.on_bar(mk(107.5, 105.5, 9))
    assert st.state is State.PULLBACK

    # Instead of another LH, we get a bar with new high > peak → back to UPTREND
    sig = st.on_bar(mk(109, 107, 10))
    assert sig is None
    assert st.state is State.UPTREND
    assert st.peak_high == 109
    print("✓ pullback aborts on new high")


def test_position_size():
    assert position_size(100_000, 0.005, 10.00, 9.50) == 1000  # risk 500 / stop 0.50 = 1000
    assert position_size(100_000, 0.005, 10.00, 9.99) == 5000  # min stop 0.10 enforces
    assert position_size(100_000, 0.005, 10.00, 10.00) == 5000  # zero diff → uses min 0.10
    assert position_size(0, 0.005, 10.00, 9.50) == 0
    print("✓ position sizing")


if __name__ == "__main__":
    test_happy_path_arm_then_fill_then_exit()
    test_reset_on_pullback_breach_while_armed()
    test_pullback_aborts_if_new_high()
    test_position_size()
    print("\nall tests passed")
