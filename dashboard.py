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
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
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
def fetch_orders(days: int = 7):
    c = alpaca_client()
    after = datetime.now(EASTERN) - timedelta(days=days)
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=after.isoformat(),
        limit=500,
    )
    orders = c.get_orders(req)
    rows = []
    for o in orders:
        rows.append({
            "submitted_at": o.submitted_at.astimezone(EASTERN) if o.submitted_at else None,
            "filled_at": o.filled_at.astimezone(EASTERN) if o.filled_at else None,
            "symbol": o.symbol,
            "side": str(o.side).split(".")[-1].lower(),
            "type": str(o.order_type).split(".")[-1].lower(),
            "qty": float(o.qty) if o.qty else 0,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "status": str(o.status).split(".")[-1].lower(),
            "order_class": str(o.order_class).split(".")[-1].lower() if o.order_class else "",
        })
    return rows


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
    st.subheader("Orders — Last 7 Days")
    rows = fetch_orders(days=7)
    if not rows:
        st.info("No orders found in the last 7 days.")
    else:
        df = pd.DataFrame(rows)
        df["submitted_at"] = df["submitted_at"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if x else "")
        df["filled_at"] = df["filled_at"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if x else "")
        filled = df[df["status"] == "filled"].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total orders", len(df))
        c2.metric("Filled", len(filled))
        c3.metric("Open", (df["status"].isin(["new", "accepted", "held", "partially_filled"])).sum())

        st.dataframe(
            df[["submitted_at", "symbol", "side", "type", "qty", "filled_qty",
                "filled_avg_price", "status", "order_class"]],
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
