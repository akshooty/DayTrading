"""Streamlit dashboard for the day-trading bot.

Surfaces simulated (backtested) trades from reports/simulated_trades.json.
Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EASTERN = ZoneInfo("America/New_York")
SIM_PATH = "reports/simulated_trades.json"

st.set_page_config(page_title="Breakout Trader", page_icon="📈", layout="wide")


@st.cache_data(ttl=60)
def load_simulated_trades() -> list[dict]:
    if not os.path.exists(SIM_PATH):
        return []
    with open(SIM_PATH) as f:
        data = json.load(f)
    for t in data:
        if t.get("entry_at"):
            t["entry_at"] = datetime.fromisoformat(t["entry_at"])
        if t.get("exit_at"):
            t["exit_at"] = datetime.fromisoformat(t["exit_at"])
    return data


@st.cache_data(ttl=300)
def fetch_scan():
    from scanner import scan_gainers, scan_losers
    return scan_gainers(), scan_losers()


# ================== UI ==================

st.title("📈 Breakout Trader Dashboard")

trades = load_simulated_trades()

if not trades:
    st.warning(
        f"No simulated trades found at `{SIM_PATH}`. "
        "Run `FEED=SIP python simulate_trades.py 10` to generate the last 10 trading days."
    )
    st.stop()

df = pd.DataFrame(trades)
closed = df[df["status"] == "closed"].copy().sort_values("entry_at")
dates = sorted({t["entry_at"].date() for t in trades if t.get("entry_at")})

# Top summary metrics
total_pnl = closed["pnl"].sum() if not closed.empty else 0
wins = closed[closed["pnl"] > 0]
losses = closed[closed["pnl"] < 0]
win_rate = (len(wins) / len(closed) * 100) if len(closed) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Simulated trades", len(df), f"{len(closed)} closed · {len(df) - len(closed)} open EOD")
c2.metric("Realized P&L", f"${total_pnl:+,.2f}")
c3.metric("Win rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
c4.metric("Date range", f"{len(dates)} days", f"{dates[0]} → {dates[-1]}" if dates else "")

st.divider()

tab_trades, tab_scanner, tab_insights = st.tabs(["📊 Trades", "🔍 Scanner", "💡 Insights"])

# ----- TRADES TAB -----
with tab_trades:
    # Filters
    f1, f2, f3 = st.columns([2, 2, 3])
    symbols = sorted(df["symbol"].unique().tolist())
    directions = sorted(df["direction"].unique().tolist())
    strategies = sorted(df.get("strategy", pd.Series(dtype=str)).dropna().unique().tolist())

    sym_filter = f1.multiselect("Symbol", symbols, default=[])
    dir_filter = f2.multiselect("Direction", directions, default=[])
    strat_filter = f3.multiselect("Strategy", strategies, default=[]) if strategies else []

    filtered = df.copy()
    if sym_filter:
        filtered = filtered[filtered["symbol"].isin(sym_filter)]
    if dir_filter:
        filtered = filtered[filtered["direction"].isin(dir_filter)]
    if strat_filter:
        filtered = filtered[filtered["strategy"].isin(strat_filter)]

    f_closed = filtered[filtered["status"] == "closed"].copy().sort_values("entry_at")

    # Detail metrics on filtered set
    if not f_closed.empty:
        f_pnl = f_closed["pnl"].sum()
        f_wins = f_closed[f_closed["pnl"] > 0]
        f_losses = f_closed[f_closed["pnl"] < 0]
        avg_win = f_wins["pnl"].mean() if not f_wins.empty else 0
        avg_loss = f_losses["pnl"].mean() if not f_losses.empty else 0
        largest_win = f_wins["pnl"].max() if not f_wins.empty else 0
        largest_loss = f_losses["pnl"].min() if not f_losses.empty else 0
        avg_duration = f_closed["duration_min"].mean()

        cum = f_closed["pnl"].cumsum()
        running_peak = cum.cummax()
        drawdown = cum - running_peak
        max_dd = drawdown.min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filtered P&L", f"${f_pnl:+,.2f}", f"{len(f_closed)} closed")
        m2.metric("Max drawdown", f"${max_dd:,.2f}")
        m3.metric("Avg win / loss", f"${avg_win:+,.2f} / ${avg_loss:+,.2f}")
        m4.metric("Avg duration", f"{avg_duration:.1f} min")

        m5, m6, m7, _ = st.columns(4)
        m5.metric("Largest win", f"${largest_win:+,.2f}")
        m6.metric("Largest loss", f"${largest_loss:+,.2f}")
        m7.metric("Win rate (filtered)", f"{(len(f_wins)/len(f_closed)*100):.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=f_closed["exit_at"], y=cum.values,
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=f_closed["exit_at"], y=drawdown.values,
            mode="lines", name="Drawdown",
            line=dict(color="#d62728", dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="Cumulative P&L ($)"),
            yaxis2=dict(title="Drawdown ($)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily P&L bar chart
        daily = f_closed.copy()
        daily["day"] = daily["exit_at"].dt.date
        by_day = daily.groupby("day")["pnl"].sum().reset_index()
        bar = go.Figure(go.Bar(
            x=by_day["day"], y=by_day["pnl"],
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in by_day["pnl"]],
        ))
        bar.update_layout(
            height=240, margin=dict(l=0, r=0, t=30, b=0),
            title="Daily realized P&L",
            yaxis_title="P&L ($)",
        )
        st.plotly_chart(bar, use_container_width=True)

    # Per-trade table
    st.subheader("Round-trip trades")
    disp = filtered.copy()
    disp["entry"] = disp.apply(
        lambda r: f"{r['entry_at'].strftime('%m-%d %H:%M')} @ ${r['entry_price']:.2f}"
        if pd.notna(r["entry_at"]) else "", axis=1,
    )
    disp["exit"] = disp.apply(
        lambda r: f"{r['exit_at'].strftime('%m-%d %H:%M')} @ ${r['exit_price']:.2f}"
        if pd.notna(r["exit_at"]) and pd.notna(r["exit_price"]) else "—",
        axis=1,
    )
    disp["duration"] = disp["duration_min"].apply(lambda x: f"{x:.0f}m" if pd.notna(x) else "—")
    disp["pnl_str"] = disp["pnl"].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "—")

    cols = ["symbol", "direction", "qty", "entry", "exit", "duration", "pnl_str", "status"]
    if "strategy" in disp.columns:
        cols.insert(2, "strategy")

    st.dataframe(
        disp[cols].rename(columns={"pnl_str": "P&L"}).sort_values("entry", ascending=False),
        use_container_width=True, hide_index=True,
    )

    # Download button — raw JSON so users can do their own analysis
    with open(SIM_PATH, "rb") as f:
        st.download_button(
            "📥 Download raw simulated_trades.json",
            f.read(), file_name="simulated_trades.json", mime="application/json",
        )

# ----- SCANNER TAB -----
with tab_scanner:
    st.subheader("Today's Universe")
    col_a, _ = st.columns([1, 5])
    if col_a.button("🔄 Refresh scan", help="Calls Finviz; takes ~5 sec"):
        st.cache_data.clear()
        st.rerun()

    try:
        with st.spinner("Scanning Finviz..."):
            gainers, losers = fetch_scan()
    except Exception as e:
        st.error(f"Scanner failed: {e}")
        gainers, losers = [], []

    g_col, l_col = st.columns(2)
    with g_col:
        st.subheader(f"🟢 Gainers ({len(gainers)})")
        if gainers:
            gdf = pd.DataFrame(gainers)
            if "Change" in gdf.columns:
                gdf["Change%"] = (gdf["Change"] * 100).apply(lambda x: f"{x:+.1f}%")
            cols = [c for c in ["Ticker", "Price", "Change%", "Volume", "Float", "Short Ratio"] if c in gdf.columns]
            st.dataframe(gdf[cols], use_container_width=True, hide_index=True)
    with l_col:
        st.subheader(f"🔴 Losers ({len(losers)})")
        if losers:
            ldf = pd.DataFrame(losers)
            if "Change" in ldf.columns:
                ldf["Change%"] = (ldf["Change"] * 100).apply(lambda x: f"{x:+.1f}%")
            cols = [c for c in ["Ticker", "Price", "Change%", "Volume", "Float", "Short Ratio"] if c in ldf.columns]
            st.dataframe(ldf[cols], use_container_width=True, hide_index=True)

# ----- INSIGHTS TAB -----
with tab_insights:
    st.subheader("Backtest Summary")
    report_path = "reports/year_analysis_2026-04-17.md"
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("No backtest report found. Run `python backtest_week.py 22` to generate history.")

    st.divider()
    st.subheader("Strategy Configuration")
    st.code("""
Entry signals:
  - BreakoutLong:  uptrend (4 HH bars) → pullback (2 LH bars) → reversal (1 HH) → break HOD
  - BreakoutShort: downtrend (4 LL bars) → bounce (2 HL bars) → reversal (1 LL) → break LOD
  - ORB:           first 10-min range → break in either direction

Risk management:
  - Max risk per trade:       $200 (hard)
  - Max deployment per trade: $2,500
  - Max concurrent positions: 16
  - Hard take-profit at:      +7% (long) / -7% (short)
  - Initial stop:             at pullback_low / bounce_high / range opposite

Guardrails:
  - Skip new entries during 11:30 AM - 2:00 PM ET (chop window)
  - Circuit breaker: stop new entries if day P&L <= -1% equity
  - EOD force-close at 2:55 PM CT
""", language="yaml")

st.divider()
st.caption(
    f"Last refreshed: {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} · "
    f"Source: {SIM_PATH} ({len(trades)} trades) · "
    "Regenerate with `FEED=SIP python simulate_trades.py N`."
)
