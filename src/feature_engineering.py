"""
Feature engineering module for stock price prediction.
Creates technical indicators and other features from raw price data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_moving_averages(df, windows=[5, 10, 20, 50, 200]):
    """
    Add simple moving averages for the specified windows.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    windows : list
        List of window sizes for moving averages

    Returns:
    --------
    pandas.DataFrame
        DataFrame with moving average columns added
    """
    df_copy = df.copy()

    for window in windows:
        df_copy[f'SMA_{window}'] = df_copy['Close'].rolling(window=window).mean()

    return df_copy


def add_exponential_moving_averages(df, windows=[5, 10, 20, 50, 200]):
    """
    Add exponential moving averages for the specified windows.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    windows : list
        List of window sizes for exponential moving averages

    Returns:
    --------
    pandas.DataFrame
        DataFrame with exponential moving average columns added
    """
    df_copy = df.copy()

    for window in windows:
        df_copy[f'EMA_{window}'] = df_copy['Close'].ewm(span=window, adjust=False).mean()

    return df_copy


def add_rsi(df, window=14):
    """
    Add Relative Strength Index (RSI).

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    window : int
        Window size for calculating RSI

    Returns:
    --------
    pandas.DataFrame
        DataFrame with RSI column added
    """
    df_copy = df.copy()

    # Calculate price differences
    delta = df_copy['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate relative strength
    rs = avg_gain / avg_loss

    # Calculate RSI
    df_copy[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    return df_copy


def add_macd(df, fast=12, slow=26, signal=9):
    """
    Add Moving Average Convergence Divergence (MACD).

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    fast : int
        Fast EMA window
    slow : int
        Slow EMA window
    signal : int
        Signal line window

    Returns:
    --------
    pandas.DataFrame
        DataFrame with MACD columns added
    """
    df_copy = df.copy()

    # Calculate fast and slow EMAs
    fast_ema = df_copy['Close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df_copy['Close'].ewm(span=slow, adjust=False).mean()

    # Calculate MACD line
    df_copy['MACD_Line'] = fast_ema - slow_ema

    # Calculate signal line
    df_copy['MACD_Signal'] = df_copy['MACD_Line'].ewm(span=signal, adjust=False).mean()

    # Calculate MACD histogram
    df_copy['MACD_Histogram'] = df_copy['MACD_Line'] - df_copy['MACD_Signal']

    return df_copy


def add_bollinger_bands(df, window=20, num_std=2):
    """
    Add Bollinger Bands.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    window : int, default=20
        Window size for calculating Bollinger Bands
    num_std : int, default=2
        Number of standard deviations for the bands

    Returns:
    --------
    pandas.DataFrame
        DataFrame with Bollinger Bands columns added
    """
    df_copy = df.copy()

    # Calculate middle band (simple moving average)
    df_copy['BB_Middle'] = df_copy['Close'].rolling(window=window).mean()

    # Calculate standard deviation
    rolling_std = df_copy['Close'].rolling(window=window).std()

    # Calculate upper and lower bands
    df_copy['BB_Upper'] = df_copy['BB_Middle'] + (rolling_std * num_std)
    df_copy['BB_Lower'] = df_copy['BB_Middle'] - (rolling_std * num_std)

    return df_copy

def add_volatility(df, windows=[5, 10, 20]):
    """
    Add volatility metrics.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    windows : list
        List of window sizes for volatility calculation

    Returns:
    --------
    pandas.DataFrame
        DataFrame with volatility columns added
    """
    df_copy = df.copy()

    # Calculate daily returns
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()

    for window in windows:
        # Standard deviation of returns (volatility)
        df_copy[f'Volatility_{window}'] = df_copy['Daily_Return'].rolling(window=window).std()

    return df_copy


def add_target_variables(df, forward_periods=[1, 5, 10]):
    """
    Add target variables for supervised learning.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with 'Close' column
    forward_periods : list
        List of forward periods to predict

    Returns:
    --------
    pandas.DataFrame
        DataFrame with target columns added
    """
    df_copy = df.copy()

    for period in forward_periods:
        # Future price
        df_copy[f'Future_Close_{period}'] = df_copy['Close'].shift(-period)

        # Price change
        df_copy[f'Price_Change_{period}'] = df_copy[f'Future_Close_{period}'] - df_copy['Close']

        # Percentage change
        df_copy[f'Pct_Change_{period}'] = (df_copy[f'Future_Close_{period}'] / df_copy['Close'] - 1) * 100

        # Binary target (1 if price goes up, 0 if it goes down)
        df_copy[f'Target_Direction_{period}'] = (df_copy[f'Price_Change_{period}'] > 0).astype(int)

    return df_copy


def generate_features(df, target_periods=[1, 5, 10], drop_na=True):
    """
    Generate all technical indicators and features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with OHLC columns
    target_periods : list
        List of forward periods for target variables
    drop_na : bool
        Whether to drop rows with NA values

    Returns:
    --------
    pandas.DataFrame
        DataFrame with all features added
    """
    print("Generating technical indicators and features...")

    # Make a copy of the input dataframe
    df_features = df.copy()

    # Add all indicators
    df_features = add_moving_averages(df_features)
    df_features = add_exponential_moving_averages(df_features)
    df_features = add_rsi(df_features)
    df_features = add_macd(df_features)
    df_features = add_bollinger_bands(df_features)
    df_features = add_volatility(df_features)

    # Add target variables
    df_features = add_target_variables(df_features, forward_periods=target_periods)

    # Drop rows with NA values
    if drop_na:
        original_len = len(df_features)
        df_features = df_features.dropna()
        print(f"Dropped {original_len - len(df_features)} rows with NA values")

    print(f"Generated {len(df_features.columns) - len(df.columns)} new features")

    return df_features


def plot_features(df, ticker):
    """
    Plot some of the generated features.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    ticker : str
        Stock ticker symbol for title
    """
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot 1: Price and Moving Averages
    axs[0].plot(df['Date'], df['Close'], label='Close Price')
    axs[0].plot(df['Date'], df['SMA_20'], label='SMA 20')
    axs[0].plot(df['Date'], df['SMA_50'], label='SMA 50')
    axs[0].plot(df['Date'], df['SMA_200'], label='SMA 200')
    axs[0].set_title(f'{ticker} - Price and Moving Averages')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: RSI
    axs[1].plot(df['Date'], df['RSI_14'], label='RSI 14')
    axs[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axs[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axs[1].set_title(f'{ticker} - Relative Strength Index')
    axs[1].set_ylabel('RSI')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: MACD
    axs[2].plot(df['Date'], df['MACD_Line'], label='MACD Line')
    axs[2].plot(df['Date'], df['MACD_Signal'], label='Signal Line')
    axs[2].bar(df['Date'], df['MACD_Histogram'], label='Histogram', alpha=0.3)
    axs[2].set_title(f'{ticker} - MACD')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('MACD')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'data/{ticker}_features.png')
    print(f"Chart saved to data/{ticker}_features.png")
    plt.show()


if __name__ == "__main__":
    # Test the feature engineering functions
    import os
    from data_collection import fetch_stock_data

    # Example usage with a specific ticker
    ticker = 'AAPL'

    # Check if we already have data
    csv_path = f'data/{ticker}_' + pd.Timestamp.now().strftime('%Y%m%d') + '.csv'
    print(f"Looking for file: {csv_path}")

    if os.path.exists(csv_path):
        print(f"Loading existing data from {csv_path}")

        # Read the CSV with the correct structure
        # Skip the second row which contains 'AAPL' values
        stock_data = pd.read_csv(csv_path, skiprows=[1])

        # Print data info to verify
        print("\nData shape:", stock_data.shape)
        print("Columns:", stock_data.columns.tolist())
        print("\nFirst 5 rows of data:")
        print(stock_data.head())

        # Ensure Date column is datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # Convert columns to numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

        # Drop any rows with NaN values
        original_len = len(stock_data)
        stock_data = stock_data.dropna()
        if len(stock_data) < original_len:
            print(f"Dropped {original_len - len(stock_data)} rows with NaN values")

        print("\nData types after cleaning:")
        print(stock_data.dtypes)
    else:
        print(f"File not found: {csv_path}")
        print(f"Fetching new data for {ticker}")
        stock_data = fetch_stock_data(ticker, period="1y")

    print("\nProceeding with feature generation...")

    # Generate features
    try:
        features_df = generate_features(stock_data)

        # Display feature names
        print("\nGenerated features:")
        for col in features_df.columns:
            if col not in stock_data.columns:
                print(f"- {col}")

        # Display some sample data
        print("\nSample data with features (last 5 rows):")
        print(features_df.tail().to_string())

        # Save the features to CSV
        features_csv = f'data/{ticker}_features_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
        features_df.to_csv(features_csv, index=False)
        print(f"\nSaved features to {features_csv}")

        # Plot some features
        try:
            plot_features(features_df, ticker)
            print("Generated plots successfully")
        except Exception as e:
            print(f"Error creating plot: {e}")
    except Exception as e:
        print(f"Error generating features: {e}")
        import traceback

        traceback.print_exc()