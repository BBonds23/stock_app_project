# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily data from Yahoo Finance for a given date range."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

# -- Main logic -------------------------------------------
if ticker:
    try:
        df = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    if df.empty:
        st.error(
            f"No data found for **{ticker}**. "
            "Check the ticker symbol and try again."
        )
        st.stop()

    # Flatten any multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -- Compute a derived column -------------------------
    df["Daily Return"] = df["Close"].pct_change()

    # -- Key metrics --------------------------------------
    latest_close = float(df["Close"].iloc[-1])
    total_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
    volatility = float(df["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)  # Annualize: daily sigma * sqrt(trading days)
    max_close = float(df["Close"].max())
    min_close = float(df["Close"].min())

    st.subheader(f"{ticker} — Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close", f"${latest_close:,.2f}")
    col2.metric("1-Year Return", f"{total_return:.2%}")
    col3.metric("Annualized Volatility (sigma)", f"{ann_volatility:.2%}")

    col4, col5, _ = st.columns(3)
    col4.metric("Period High", f"${max_close:,.2f}")
    col5.metric("Period Low", f"${min_close:,.2f}")

    st.divider()

    # -- Price chart --------------------------------------
    st.subheader("Closing Price")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Close Price",
            line=dict(width=1.5)
        )
    )
    fig.update_layout(
        yaxis_title="Price (USD)", xaxis_title="Date",
        template="plotly_white", height=450
    )
    st.plotly_chart(fig, width="stretch")

else:
    st.info("Enter a stock ticker in the sidebar to get started.")