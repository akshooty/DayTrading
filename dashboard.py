"""Streamlit dashboard for the day-trading bot.

Run locally:    streamlit run dashboard.py
Open:           http://localhost:8501

For remote / online hosting, see README (Railway, Render, Fly.io).
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv

EASTERN = ZoneInfo("America/New_York")

load_dotenv()
st.set_page_config(page_title="Breakout Trader", page_icon="📈", layout="wide")


@st.cache_resource
def alpaca_client() -> TradingClient:
    return TradingClient(
        os.environ["ALPACA_API_KEY"].strip(),
        os.environ["ALPACA_SECRET_KEY"].strip(),
        paper=True,
    )


@st.cache_data(ttl=30)
def fetch_account():
    c = alpaca_client()
    acct = c.get_account()
    return {
        "equity": float(acct.equity),
        "last_equity": float(acct.last_equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "status": str(acct.status),
        "account_number": acct.account_number,
        "pattern_day_trader": acct.pattern_day_trader,
    }


@st.cache_data(ttl=30)
def fetch_positions():
    c = alpaca_client()
    positions = c.get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "side": str(p.side).split(".")[-1].lower(),
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current": float(p.current_price) if p.current_price else 0,
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_pct": float(p.unrealized_plpc) * 100,
        }
        for p in positions
    ]


@st.cache_data(ttl=30)
def fetch_trades(days: int = 30):
    """Pair filled orders into round-trip trades (entry + exit) and
    compute per-trade P&L, duration, and cumulative stats.

    Assumes at most one open position per symbol at any time (which is
    the bot's design invariant). For positions still open, the trade is
    returned with exit fields blank.
    """
    c = alpaca_client()
    after = datetime.now(EASTERN) - timedelta(days=days)
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=after.isoformat(),
        limit=500,
    )
    orders = c.get_orders(req)
    filled = [
        o for o in orders
        if str(o.status).split(".")[-1].lower() == "filled" and o.filled_at
    ]
    filled.sort(key=lambda o: o.filled_at)

    trades = []
    open_pos: dict[str, dict] = {}

    for o in filled:
        sym = o.symbol
        side = str(o.side).split(".")[-1].lower()
        qty = float(o.filled_qty or 0)
        price = float(o.filled_avg_price or 0)
        t = o.filled_at.astimezone(EASTERN)

        if sym not in open_pos:
            direction = "long" if side.endswith("buy") else "short"
            open_pos[sym] = {
                "entry_at": t, "entry_price": price, "qty": qty,
                "direction": direction,
            }
        else:
            ent = open_pos.pop(sym)
            if ent["direction"] == "long":
                pnl = (price - ent["entry_price"]) * min(qty, ent["qty"])
            else:
                pnl = (ent["entry_price"] - price) * min(qty, ent["qty"])
            trades.append({
                "entry_at": ent["entry_at"],
                "exit_at": t,
                "symbol": sym,
                "direction": ent["direction"],
                "qty": ent["qty"],
                "entry_price": ent["entry_price"],
                "exit_price": price,
                "pnl": pnl,
                "duration_min": (t - ent["entry_at"]).total_seconds() / 60,
                "status": "closed",
            })

    for sym, ent in open_pos.items():
        trades.append({
            "entry_at": ent["entry_at"],
            "exit_at": None,
            "symbol": sym,
            "direction": ent["direction"],
            "qty": ent["qty"],
            "entry_price": ent["entry_price"],
            "exit_price": None,
            "pnl": None,
            "duration_min": None,
            "status": "open",
        })
    return trades


@st.cache_data(ttl=300)
def fetch_scan():
    from scanner import scan_gainers, scan_losers
    gainers = scan_gainers()
    losers = scan_losers()
    return gainers, losers


# ================== UI ==================

st.title("📈 Breakout Trader Dashboard")

acct = fetch_account()
day_pnl = acct["equity"] - acct["last_equity"]
day_pct = (day_pnl / acct["last_equity"] * 100) if acct["last_equity"] else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Account Equity", f"${acct['equity']:,.2f}", f"${day_pnl:+,.2f} today")
c2.metric("Cash", f"${acct['cash']:,.2f}")
c3.metric("Buying Power", f"${acct['buying_power']:,.2f}")
c4.metric("Day Return", f"{day_pct:+.3f}%", f"{acct['account_number']} · paper")

if acct["pattern_day_trader"]:
    st.info("Account is flagged as **Pattern Day Trader**.")

st.divider()

tab_trades, tab_positions, tab_scanner, tab_insights = st.tabs(
    ["📊 Trades", "💼 Positions", "🔍 Scanner", "💡 Insights"]
)

# ----- TRADES TAB -----
with tab_trades:
    days_back = st.slider("Lookback (days)", 1, 90, 30, 1)
    trades = fetch_trades(days=days_back)
    if not trades:
        st.info("No trades found in this window. Once the bot runs live (DRY_RUN=0), round-trip trades will show here.")
    else:
        df = pd.DataFrame(trades)
        closed = df[df["status"] == "closed"].copy()
        closed = closed.sort_values("entry_at")

        # Summary metrics
        total_pnl = closed["pnl"].sum() if not closed.empty else 0
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] < 0]
        win_rate = (len(wins) / len(closed) * 100) if len(closed) else 0
        avg_win = wins["pnl"].mean() if not wins.empty else 0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0
        largest_win = wins["pnl"].max() if not wins.empty else 0
        largest_loss = losses["pnl"].min() if not losses.empty else 0
        avg_duration = closed["duration_min"].mean() if not closed.empty else 0

        # Drawdown from cumulative P&L
        if not closed.empty:
            cum = closed["pnl"].cumsum()
            running_peak = cum.cummax()
            drawdown = cum - running_peak
            max_dd = drawdown.min()
        else:
            cum = pd.Series(dtype=float)
            max_dd = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Closed trades", len(closed))
        c2.metric("Realized P&L", f"${total_pnl:+,.2f}")
        c3.metric("Win rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
        c4.metric("Max drawdown", f"${max_dd:,.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg win", f"${avg_win:+,.2f}")
        c6.metric("Avg loss", f"${avg_loss:+,.2f}")
        c7.metric("Largest win", f"${largest_win:+,.2f}")
        c8.metric("Avg duration", f"{avg_duration:.1f} min")

        # Equity curve (cumulative realized P&L)
        if not closed.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed["exit_at"],
                y=cum.values,
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#1f77b4"),
            ))
            fig.add_trace(go.Scatter(
                x=closed["exit_at"],
                y=drawdown.values,
                mode="lines",
                name="Drawdown",
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

        # Per-trade table
        st.subheader("Round-trip trades")
        open_count = (df["status"] == "open").sum()
        if open_count:
            st.caption(f"{open_count} currently-open position(s) also listed below with blank exit fields.")

        disp = df.copy()
        disp["entry"] = disp.apply(
            lambda r: f"{r['entry_at'].strftime('%m-%d %H:%M')} @ ${r['entry_price']:.2f}"
            if pd.notna(r["entry_at"]) else "", axis=1,
        )
        disp["exit"] = disp.apply(
            lambda r: f"{r['exit_at'].strftime('%m-%d %H:%M')} @ ${r['exit_price']:.2f}"
            if pd.notna(r["exit_at"]) and pd.notna(r["exit_price"]) else "—",
            axis=1,
        )
        disp["duration"] = disp["duration_min"].apply(
            lambda x: f"{x:.0f}m" if pd.notna(x) else "—"
        )
        disp["pnl_str"] = disp["pnl"].apply(
            lambda x: f"${x:+,.2f}" if pd.notna(x) else "—"
        )

        st.dataframe(
            disp[["symbol", "direction", "qty", "entry", "exit", "duration", "pnl_str", "status"]]
                .rename(columns={"pnl_str": "P&L"})
                .sort_values("entry", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ----- POSITIONS TAB -----
with tab_positions:
    st.subheader("Open Positions")
    positions = fetch_positions()
    if not positions:
        st.info("No open positions.")
    else:
        df = pd.DataFrame(positions)
        total_mv = df["market_value"].sum()
        total_upl = df["unrealized_pl"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Positions", len(df))
        c2.metric("Market value", f"${total_mv:,.2f}")
        c3.metric("Unrealized P&L", f"${total_upl:+,.2f}")

        df_display = df.copy()
        df_display["unrealized_pct"] = df_display["unrealized_pct"].apply(lambda x: f"{x:+.2f}%")
        df_display["avg_entry"] = df_display["avg_entry"].apply(lambda x: f"${x:.2f}")
        df_display["current"] = df_display["current"].apply(lambda x: f"${x:.2f}")
        df_display["market_value"] = df_display["market_value"].apply(lambda x: f"${x:,.2f}")
        df_display["unrealized_pl"] = df_display["unrealized_pl"].apply(lambda x: f"${x:+,.2f}")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ----- SCANNER TAB -----
with tab_scanner:
    st.subheader("Today's Universe")
    col_a, col_b = st.columns([1, 5])
    if col_a.button("🔄 Refresh scan", help="Calls Finviz; takes ~5 sec"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Scanning Finviz..."):
        gainers, losers = fetch_scan()

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
            content = f.read()
        st.markdown(content)
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
    f"Last refreshed: {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')}. "
    f"Data cache: 30s (account/orders/positions), 5min (scanner)."
)
