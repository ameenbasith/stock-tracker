"""
Stock Tracker Streamlit Application - Enhanced Version
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Define functions to generate technical indicators
def add_moving_averages(df, windows=[20, 50, 200]):
    """Add simple moving averages"""
    df_copy = df.copy()
    for window in windows:
        df_copy[f'SMA_{window}'] = df_copy['Close'].rolling(window=window).mean()
    return df_copy


def add_rsi(df, window=14):
    """Add Relative Strength Index"""
    df_copy = df.copy()
    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df_copy[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return df_copy


# Define a function to fetch data directly with yfinance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y"):
    """Fetch stock data directly with yfinance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        data = data.reset_index()  # Reset index to make Date a column
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def plot_stock_price(data, ticker):
    """Create a matplotlib chart for stock prices"""
    # Make sure data is sorted by date
    data = data.sort_values('Date')

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price')

    # Add moving averages if they exist
    for ma in [20, 50, 200]:
        col = f'SMA_{ma}'
        if col in data.columns:
            ax.plot(data['Date'], data[col], label=f'SMA {ma}')

    # Add labels and title
    ax.set_title(f'{ticker} Stock Price and Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter('${x:,.2f}')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Show the chart
    st.pyplot(fig)


def plot_rsi(data, ticker):
    """Plot RSI indicator"""
    if 'RSI_14' not in data.columns:
        return

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Date'], data['RSI_14'], color='purple')

    # Add overbought/oversold lines
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)

    # Add labels
    ax.text(data['Date'].iloc[-1], 70, 'Overbought', verticalalignment='bottom', horizontalalignment='right')
    ax.text(data['Date'].iloc[-1], 30, 'Oversold', verticalalignment='top', horizontalalignment='right')

    # Add labels and title
    ax.set_title(f'{ticker} - Relative Strength Index (RSI)')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Show the chart
    st.pyplot(fig)


def predict_price_movement(data, ticker, days_ahead=1):
    """Make a simple prediction based on recent trends - for demo purposes only"""
    # Calculate average price change over the last 5 days
    recent_data = data.tail(10)
    avg_change = recent_data['Close'].pct_change().mean() * 100

    # Convert to float to avoid Series comparison error
    avg_change = float(avg_change)

    # Get current price
    current_price = float(data['Close'].iloc[-1])

    # Calculate predicted price
    predicted_change = avg_change * days_ahead
    predicted_price = current_price * (1 + predicted_change / 100)

    # Determine trend direction
    if avg_change > 0.5:
        trend = "Strong Uptrend"
        emoji = "🚀"
    elif avg_change > 0:
        trend = "Mild Uptrend"
        emoji = "📈"
    elif avg_change > -0.5:
        trend = "Mild Downtrend"
        emoji = "📉"
    else:
        trend = "Strong Downtrend"
        emoji = "🔻"

    return {
        'current_price': current_price,
        'predicted_change': predicted_change,
        'predicted_price': predicted_price,
        'trend': trend,
        'emoji': emoji
    }

# App title
st.title('📈 Stock Price Predictor')
st.markdown("""
This app analyzes stock price data, displays technical indicators, and provides trend predictions.
""")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

# Stock ticker input
default_ticker = 'AAPL'
ticker = st.sidebar.text_input('Stock Ticker', default_ticker).upper()

# Time period selection
period_options = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}
selected_period = st.sidebar.selectbox('Select Time Period', list(period_options.keys()))
period = period_options[selected_period]

# Technical indicator options
show_ma = st.sidebar.checkbox('Show Moving Averages', value=True)
show_rsi = st.sidebar.checkbox('Show RSI', value=True)

# Fetch data button
fetch_data = st.sidebar.button('Fetch Data')

if fetch_data or 'stock_data' not in st.session_state:
    # Fetch the data
    st.session_state.stock_data = get_stock_data(ticker, period=period)

    if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
        # Generate technical indicators
        if show_ma:
            st.session_state.stock_data = add_moving_averages(st.session_state.stock_data)
        if show_rsi:
            st.session_state.stock_data = add_rsi(st.session_state.stock_data)
    else:
        st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")

# Show data and make predictions
if 'stock_data' in st.session_state and st.session_state.stock_data is not None:
    # Show recent statistics
    st.header('Recent Stock Data')
    recent_data = st.session_state.stock_data.tail()
    st.dataframe(recent_data)

    # Summary statistics
    st.subheader('Summary Statistics')
    col1, col2, col3, col4 = st.columns(4)

    # Calculate some basic stats
    current_price = float(st.session_state.stock_data['Close'].iloc[-1])
    prev_price = float(st.session_state.stock_data['Close'].iloc[-2])
    pct_change = ((current_price / prev_price) - 1) * 100
    high_52wk = float(st.session_state.stock_data['High'].max())
    low_52wk = float(st.session_state.stock_data['Low'].min())

    col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:.2f}%")
    col2.metric("52-Week High", f"${high_52wk:.2f}", f"{((current_price / high_52wk) - 1) * 100:.2f}%")
    col3.metric("52-Week Low", f"${low_52wk:.2f}", f"{((current_price / low_52wk) - 1) * 100:.2f}%")
    col4.metric("Volume", f"{int(st.session_state.stock_data['Volume'].iloc[-1]):,}")

    # Plot stock price
    st.header('Price Chart')
    plot_stock_price(st.session_state.stock_data, ticker)

    # Show RSI if enabled
    if show_rsi and 'RSI_14' in st.session_state.stock_data.columns:
        st.header('Technical Indicators')
        plot_rsi(st.session_state.stock_data, ticker)

    # Make prediction
    st.header('Price Prediction')
    st.markdown("""
    *This is a simplified prediction for demonstration purposes only. It uses recent price trends to estimate future movement.*
    """)

    prediction = predict_price_movement(st.session_state.stock_data, ticker)

    # Display prediction
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric("Current Price", f"${prediction['current_price']:.2f}")
    pred_col2.metric("Predicted Daily Change", f"{prediction['predicted_change']:.2f}%")
    pred_col3.metric("Next Day Estimate", f"${prediction['predicted_price']:.2f}")

    # Display trend
    st.subheader(f"Market Trend: {prediction['emoji']} {prediction['trend']}")

    if abs(prediction['predicted_change']) > 2:
        st.warning("⚠️ This prediction suggests significant volatility. Use caution in your investment decisions.")

# Add a section for multi-stock comparison
st.header('Compare Multiple Stocks')
compare_tickers = st.text_input('Enter multiple tickers separated by commas (e.g., AAPL,MSFT,GOOGL)', 'AAPL,MSFT')

if st.button('Compare Stocks'):
    tickers = [ticker.strip() for ticker in compare_tickers.split(',')]

    # Fetch data for all tickers
    comparison_data = {}
    for ticker in tickers:
        data = get_stock_data(ticker, period=period)
        if data is not None and not data.empty:
            # Normalize to percentage change from first day
            first_price = data['Close'].iloc[0]
            data['Normalized'] = (data['Close'] / first_price - 1) * 100
            comparison_data[ticker] = data

    # Plot comparison chart
    if comparison_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        for ticker, data in comparison_data.items():
            ax.plot(data['Date'], data['Normalized'], label=ticker)

        # Add labels and title
        ax.set_title('Stock Price Comparison (% Change)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price Change (%)')

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Tight layout
        plt.tight_layout()

        # Show the chart
        st.pyplot(fig)
    else:
        st.error("Could not fetch data for the provided tickers.")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This application is for educational purposes only. The predictions should not be used for actual trading decisions.
""")
