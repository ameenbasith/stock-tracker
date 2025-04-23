"""
Data collection module for stock price prediction.
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

def fetch_stock_data(ticker, period="1y", interval="1d", max_retries=3):
    """
    Fetch historical stock data from Yahoo Finance with retry logic.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str, default="1y"
        Data period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    interval : str, default="1d"
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    max_retries : int, default=3
        Maximum number of retry attempts

    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns Open, High, Low, Close, Volume, etc.
    """
    retries = 0
    while retries < max_retries:
        try:
            print(f"Fetching data for {ticker} with period={period} and interval={interval} (Attempt {retries+1})")
            # Try downloading directly from yfinance download function
            data = yf.download(ticker, period=period, interval=interval, progress=False)

            # Check if we got any data
            if data.empty:
                print(f"No data retrieved for {ticker}, trying again...")
                retries += 1
                time.sleep(1)  # Wait a second before retrying
                continue

            # Reset index to make Date a column
            data = data.reset_index()

            print(f"Successfully fetched {len(data)} records for {ticker}")
            return data

        except Exception as e:
            print(f"Error fetching data for {ticker} (Attempt {retries+1}): {str(e)}")
            retries += 1

            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  # Wait before retrying
            else:
                print(f"Failed to fetch data for {ticker} after {max_retries} attempts")
                # Return an empty DataFrame with the expected columns
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

def save_to_csv(df, ticker, data_dir='data'):
    """
    Save the dataframe as a CSV file.

    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data to save
    ticker : str
        Stock ticker symbol
    data_dir : str, default='data'
        Directory to save the data
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Create filename with current date
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_{today}.csv"
    filepath = os.path.join(data_dir, filename)

    # Check if DataFrame is empty
    if df.empty:
        print(f"Warning: Saving empty DataFrame for {ticker}")

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

    return filepath

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    data = fetch_stock_data(ticker, period="1y")

    # Print some basic statistics to verify data
    if not data.empty:
        print("\nData summary:")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Average closing price: ${float(data['Close'].mean()):.2f}")
        print(f"Price range: ${float(data['Low'].min()):.2f} to ${float(data['High'].max()):.2f}")

    save_to_csv(data, ticker)