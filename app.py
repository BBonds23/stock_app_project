import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import jarque_bera, kurtosis, norm, probplot, skew


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Stock Comparison & Analysis App", layout="wide")
st.title("Stock Comparison & Analysis App")
st.caption("Compare 2–5 stocks, benchmark them against the S&P 500, and explore diversification with a two-asset portfolio.")

TRADING_DAYS = 252
BENCHMARK = "^GSPC"
MIN_DAYS = 365


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def parse_tickers(raw_text: str) -> list[str]:
    tickers = []
    for item in raw_text.replace("\n", ",").split(","):
        cleaned = item.strip().upper()
        if cleaned and cleaned not in tickers:
            tickers.append(cleaned)
    return tickers


@st.cache_data(ttl=3600, show_spinner=False)
def download_single_ticker(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    df = flatten_columns(df)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(tickers: tuple[str, ...], start_date: date, end_date: date):
    all_prices = {}
    failed = []
    partial = []

    for ticker in list(tickers) + [BENCHMARK]:
        try:
            df = download_single_ticker(ticker, start_date, end_date)
        except Exception:
            failed.append(ticker)
            continue

        if df.empty or "Adj Close" not in df.columns:
            failed.append(ticker)
            continue

        series = df["Adj Close"].copy().dropna()
        if series.empty:
            failed.append(ticker)
            continue

        if series.index.min().date() > start_date or series.index.max().date() < end_date:
            partial.append(ticker)

        all_prices[ticker] = series

    return all_prices, failed, partial


def align_selected_data(price_dict: dict[str, pd.Series], selected_tickers: list[str]):
    chosen = {t: price_dict[t] for t in selected_tickers if t in price_dict}
    benchmark = {BENCHMARK: price_dict[BENCHMARK]} if BENCHMARK in price_dict else {}

    if len(chosen) < 2:
        return None, None, None

    stock_prices = pd.concat(chosen, axis=1).sort_index()
    stock_prices = stock_prices.dropna(how="any")

    benchmark_prices = None
    if benchmark:
        benchmark_prices = pd.concat(benchmark, axis=1).sort_index()
        benchmark_prices = benchmark_prices.reindex(stock_prices.index).ffill().dropna()

    combined_prices = stock_prices.copy()
    if benchmark_prices is not None and not benchmark_prices.empty:
        combined_prices = combined_prices.join(benchmark_prices, how="inner")
        stock_prices = combined_prices[selected_tickers]
        benchmark_prices = combined_prices[[BENCHMARK]]

    if stock_prices.empty or len(stock_prices) < 2:
        return None, None, None

    stock_returns = stock_prices.pct_change().dropna()
    benchmark_returns = None
    all_returns = stock_returns.copy()

    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_returns = benchmark_prices.pct_change().dropna()
        all_returns = stock_returns.join(benchmark_returns, how="inner")
        stock_returns = all_returns[selected_tickers]
        benchmark_returns = all_returns[[BENCHMARK]]

    return stock_prices, benchmark_prices, stock_returns.join(benchmark_returns, how="inner") if benchmark_returns is not None else stock_returns


@st.cache_data(ttl=3600, show_spinner=False)
def compute_summary_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=returns_df.columns)
    stats["Annualized Mean Return"] = returns_df.mean() * TRADING_DAYS
    stats["Annualized Volatility"] = returns_df.std() * math.sqrt(TRADING_DAYS)
    stats["Skewness"] = returns_df.apply(skew)
    stats["Kurtosis"] = returns_df.apply(lambda x: kurtosis(x, fisher=False))
    stats["Min Daily Return"] = returns_df.min()
    stats["Max Daily Return"] = returns_df.max()
    return stats


def make_price_chart(price_df: pd.DataFrame, selected_series: list[str]) -> go.Figure:
    fig = go.Figure()
    for col in selected_series:
        if col in price_df.columns:
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode="lines", name=col))
    fig.update_layout(
        title="Adjusted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Adjusted Close Price",
        template="plotly_white",
        height=500,
    )
    return fig


def make_wealth_index(stock_returns: pd.DataFrame, benchmark_returns: pd.DataFrame | None, initial_value: int = 10000) -> pd.DataFrame:
    wealth = (1 + stock_returns).cumprod() * initial_value
    wealth["Equal-Weight Portfolio"] = (1 + stock_returns.mean(axis=1)).cumprod() * initial_value
    if benchmark_returns is not None and not benchmark_returns.empty:
        wealth[BENCHMARK] = (1 + benchmark_returns.iloc[:, 0]).cumprod() * initial_value
    return wealth


def make_rolling_vol_chart(stock_returns: pd.DataFrame, window: int) -> go.Figure:
    rolling_vol = stock_returns.rolling(window).std() * math.sqrt(TRADING_DAYS)
    fig = go.Figure()
    for col in rolling_vol.columns:
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col))
    fig.update_layout(
        title=f"Rolling Annualized Volatility ({window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        height=500,
    )
    return fig


def make_histogram_with_normal(series: pd.Series, ticker: str) -> go.Figure:
    mu, sigma = norm.fit(series)
    x_vals = np.linspace(series.min(), series.max(), 300)
    y_vals = norm.pdf(x_vals, mu, sigma)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            histnorm="probability density",
            name="Daily Returns",
            opacity=0.75,
            nbinsx=50,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="Fitted Normal Curve",
        )
    )
    fig.update_layout(
        title=f"Return Distribution for {ticker}",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        template="plotly_white",
        height=500,
        barmode="overlay",
    )
    return fig


def make_qq_plot(series: pd.Series, ticker: str) -> go.Figure:
    (theoretical_q, ordered_vals), (slope, intercept, _) = probplot(series, dist="norm")
    line_x = np.array([theoretical_q.min(), theoretical_q.max()])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=ordered_vals,
            mode="markers",
            name="Observed Quantiles",
        )
    )
    fig.add_trace(
        go.Scatter(x=line_x, y=line_y, mode="lines", name="Reference Line")
    )
    fig.update_layout(
        title=f"Q-Q Plot for {ticker}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        height=500,
    )
    return fig


def make_box_plot(stock_returns: pd.DataFrame) -> go.Figure:
    long_df = stock_returns.reset_index().melt(id_vars=stock_returns.index.name or "Date", var_name="Ticker", value_name="Daily Return")
    fig = px.box(long_df, x="Ticker", y="Daily Return", title="Box Plot of Daily Returns")
    fig.update_layout(template="plotly_white", xaxis_title="Ticker", yaxis_title="Daily Return", height=500)
    return fig


def make_corr_heatmap(stock_returns: pd.DataFrame) -> go.Figure:
    corr = stock_returns.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap of Daily Returns",
    )
    fig.update_layout(height=500)
    return fig


def make_scatter_plot(stock_returns: pd.DataFrame, stock_a: str, stock_b: str) -> go.Figure:
    fig = px.scatter(
        stock_returns,
        x=stock_a,
        y=stock_b,
                title=f"Daily Return Scatter Plot: {stock_a} vs {stock_b}",
    )
    fig.update_layout(template="plotly_white", xaxis_title=f"{stock_a} Daily Return", yaxis_title=f"{stock_b} Daily Return", height=500)
    return fig


def make_rolling_corr_plot(stock_returns: pd.DataFrame, stock_a: str, stock_b: str, window: int) -> go.Figure:
    rolling_corr = stock_returns[stock_a].rolling(window).corr(stock_returns[stock_b])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name="Rolling Correlation"))
    fig.update_layout(
        title=f"Rolling Correlation: {stock_a} vs {stock_b} ({window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=500,
    )
    return fig


def build_two_asset_portfolio(stock_returns: pd.DataFrame, stock_a: str, stock_b: str):
    pair_returns = stock_returns[[stock_a, stock_b]].dropna()
    mean_returns_annual = pair_returns.mean() * TRADING_DAYS
    cov_annual = pair_returns.cov() * TRADING_DAYS

    weights = np.linspace(0, 1, 101)
    portfolio_returns = []
    portfolio_vols = []

    for w in weights:
        portfolio_return = w * mean_returns_annual[stock_a] + (1 - w) * mean_returns_annual[stock_b]
        variance = (
            (w ** 2) * cov_annual.loc[stock_a, stock_a]
            + ((1 - w) ** 2) * cov_annual.loc[stock_b, stock_b]
            + 2 * w * (1 - w) * cov_annual.loc[stock_a, stock_b]
        )
        portfolio_volatility = math.sqrt(max(variance, 0))
        portfolio_returns.append(portfolio_return)
        portfolio_vols.append(portfolio_volatility)

    frontier = pd.DataFrame(
        {
            f"Weight in {stock_a}": weights,
            "Annualized Return": portfolio_returns,
            "Annualized Volatility": portfolio_vols,
        }
    )
    return frontier, mean_returns_annual, cov_annual


def make_portfolio_vol_chart(frontier: pd.DataFrame, stock_a: str, current_weight: float) -> go.Figure:
    x_col = f"Weight in {stock_a}"
    y_col = "Annualized Volatility"
    current_row = frontier.iloc[(frontier[x_col] - current_weight).abs().argmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier[x_col], y=frontier[y_col], mode="lines", name="Portfolio Volatility Curve"))
    fig.add_trace(
        go.Scatter(
            x=[current_row[x_col]],
            y=[current_row[y_col]],
            mode="markers",
            name="Current Weight",
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=f"Two-Asset Portfolio Volatility Curve ({stock_a} Weight)",
        xaxis_title=f"Weight on {stock_a}",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        height=500,
    )
    return fig


# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
st.sidebar.header("Input Settings")
default_end = date.today()
default_start = default_end - timedelta(days=365 * 3)

raw_ticker_input = st.sidebar.text_area(
    "Enter 2 to 5 stock tickers (comma-separated)",
    value="AAPL, MSFT, NVDA",
    help="Example: AAPL, MSFT, GOOGL",
)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=default_end)
run_analysis = st.sidebar.button("Run Analysis", type="primary")

with st.sidebar.expander("About / Methodology", expanded=False):
    st.write(
        "This app downloads adjusted closing prices using yfinance and computes simple daily returns. "
        "Annualized return uses mean daily return × 252. Annualized volatility uses daily standard deviation × √252. "
        "The S&P 500 (^GSPC) is included as a benchmark for comparison."
    )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
selected_tickers = parse_tickers(raw_ticker_input)

if not run_analysis:
    st.info("Set your tickers and date range in the sidebar, then click **Run Analysis**.")
    st.stop()

if len(selected_tickers) < 2 or len(selected_tickers) > 5:
    st.error("Please enter between 2 and 5 unique stock tickers.")
    st.stop()

if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

if (end_date - start_date).days < MIN_DAYS:
    st.error("Please choose a date range of at least 1 year.")
    st.stop()


# ------------------------------------------------------------
# Data load
# ------------------------------------------------------------
with st.spinner("Downloading market data and running analysis..."):
    price_dict, failed_tickers, partial_tickers = download_price_data(tuple(selected_tickers), start_date, end_date)

available_selected = [t for t in selected_tickers if t in price_dict]

if failed_tickers:
    st.error("These tickers failed to download or had insufficient data: " + ", ".join(failed_tickers))

if len(available_selected) < 2:
    st.error("Fewer than 2 valid stock tickers remain after validation. Please revise your inputs.")
    st.stop()

if partial_tickers:
    st.warning(
        "Some tickers did not cover the full selected period and were aligned using the overlapping available range: "
        + ", ".join(partial_tickers)
    )

stock_prices, benchmark_prices, all_returns = align_selected_data(price_dict, available_selected)

if stock_prices is None or all_returns is None or stock_prices.empty or all_returns.empty:
    st.error("Unable to build an overlapping dataset across the selected stocks. Try different tickers or a different date range.")
    st.stop()

stock_returns = all_returns[available_selected]
benchmark_returns = all_returns[[BENCHMARK]] if BENCHMARK in all_returns.columns else None
summary_stats = compute_summary_stats(all_returns)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Price & Returns",
    "Risk & Distribution",
    "Correlation & Diversification",
])


# ------------------------------------------------------------
# Tab 1: Price & Returns
# ------------------------------------------------------------
with tab1:
    st.header("Price and Return Analysis")

    chart_options = available_selected.copy()
    if benchmark_returns is not None:
        chart_options.append(BENCHMARK)

    visible_series = st.multiselect(
        "Select series to display on the price chart",
        options=chart_options,
        default=chart_options,
    )

    combined_prices_for_chart = stock_prices.copy()
    if benchmark_prices is not None:
        combined_prices_for_chart = combined_prices_for_chart.join(benchmark_prices, how="left")

    if visible_series:
        st.plotly_chart(make_price_chart(combined_prices_for_chart, visible_series), use_container_width=True)
    else:
        st.warning("Select at least one series to display the price chart.")

    st.subheader("Summary Statistics Table")
    display_stats = summary_stats.copy()
    for col in display_stats.columns:
        display_stats[col] = display_stats[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    st.dataframe(display_stats, use_container_width=True)

    st.subheader("Cumulative Wealth Index")
    wealth_df = make_wealth_index(stock_returns, benchmark_returns, initial_value=10000)
    wealth_fig = go.Figure()
    for col in wealth_df.columns:
        wealth_fig.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df[col], mode="lines", name=col))
    wealth_fig.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(wealth_fig, use_container_width=True)


# ------------------------------------------------------------
# Tab 2: Risk & Distribution
# ------------------------------------------------------------
with tab2:
    st.header("Risk and Distribution Analysis")

    vol_window = st.selectbox("Rolling volatility window", options=[30, 60, 90], index=1)
    st.plotly_chart(make_rolling_vol_chart(stock_returns, vol_window), use_container_width=True)

    selected_dist_stock = st.selectbox("Select a stock for distribution analysis", options=available_selected)
    dist_mode = st.radio("Distribution view", options=["Histogram + Normal Curve", "Q-Q Plot"], horizontal=True)

    dist_series = stock_returns[selected_dist_stock].dropna()
    jb_stat, jb_pvalue = jarque_bera(dist_series)

    col1, col2 = st.columns([3, 1])
    with col1:
        if dist_mode == "Histogram + Normal Curve":
            st.plotly_chart(make_histogram_with_normal(dist_series, selected_dist_stock), use_container_width=True)
        else:
            st.plotly_chart(make_qq_plot(dist_series, selected_dist_stock), use_container_width=True)
    with col2:
        st.metric("Jarque-Bera Statistic", f"{jb_stat:.2f}")
        st.metric("p-value", f"{jb_pvalue:.4f}")
        st.write("**Normality Decision:**")
        if jb_pvalue < 0.05:
            st.error("Rejects normality (p < 0.05)")
        else:
            st.success("Fails to reject normality (p ≥ 0.05)")

    st.plotly_chart(make_box_plot(stock_returns), use_container_width=True)


# ------------------------------------------------------------
# Tab 3: Correlation & Diversification
# ------------------------------------------------------------
with tab3:
    st.header("Correlation and Diversification")
    st.plotly_chart(make_corr_heatmap(stock_returns), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        scatter_a = st.selectbox("Scatter Plot: Stock A", options=available_selected, index=0)
    with col_b:
        scatter_b_default = 1 if len(available_selected) > 1 else 0
        scatter_b = st.selectbox("Scatter Plot: Stock B", options=available_selected, index=scatter_b_default)

    if scatter_a == scatter_b:
        st.warning("Please choose two different stocks for the scatter plot.")
    else:
        st.plotly_chart(make_scatter_plot(stock_returns, scatter_a, scatter_b), use_container_width=True)

    st.subheader("Rolling Correlation")
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        rolling_a = st.selectbox("Rolling Corr: Stock A", options=available_selected, index=0, key="roll_a")
    with col_d:
        rolling_b = st.selectbox("Rolling Corr: Stock B", options=available_selected, index=scatter_b_default, key="roll_b")
    with col_e:
        rolling_window = st.selectbox("Rolling window length", options=[30, 60, 90], index=1)

    if rolling_a == rolling_b:
        st.warning("Please choose two different stocks for the rolling correlation chart.")
    else:
        st.plotly_chart(make_rolling_corr_plot(stock_returns, rolling_a, rolling_b, rolling_window), use_container_width=True)

    st.subheader("Two-Asset Portfolio Explorer")
    col_f, col_g = st.columns(2)
    with col_f:
        port_a = st.selectbox("Portfolio Explorer: Stock A", options=available_selected, index=0, key="port_a")
    with col_g:
        port_b = st.selectbox("Portfolio Explorer: Stock B", options=available_selected, index=scatter_b_default, key="port_b")

    if port_a == port_b:
        st.warning("Please choose two different stocks for the two-asset portfolio explorer.")
    else:
        weight_a_pct = st.slider(f"Weight on {port_a} (%)", min_value=0, max_value=100, value=50, step=1)
        weight_a = weight_a_pct / 100

        frontier_df, mean_returns_annual, cov_annual = build_two_asset_portfolio(stock_returns, port_a, port_b)
        current_return = weight_a * mean_returns_annual[port_a] + (1 - weight_a) * mean_returns_annual[port_b]
        current_variance = (
            (weight_a ** 2) * cov_annual.loc[port_a, port_a]
            + ((1 - weight_a) ** 2) * cov_annual.loc[port_b, port_b]
            + 2 * weight_a * (1 - weight_a) * cov_annual.loc[port_a, port_b]
        )
        current_vol = math.sqrt(max(current_variance, 0))

        metric1, metric2 = st.columns(2)
        metric1.metric("Portfolio Annualized Return", f"{current_return:.2%}")
        metric2.metric("Portfolio Annualized Volatility", f"{current_vol:.2%}")

        st.plotly_chart(make_portfolio_vol_chart(frontier_df, port_a, weight_a), use_container_width=True)
        st.info(
            "This curve demonstrates diversification. When two stocks are not perfectly correlated, combining them can create "
            "a portfolio with lower volatility than holding either stock alone. The lower the correlation, the stronger the diversification benefit tends to be."
        )
