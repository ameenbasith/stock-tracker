"""
Simple test script to verify data collection is working.
"""
from src.data_collection import fetch_stock_data, save_to_csv

# Test with a few popular stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

for ticker in tickers:
    try:
        # Fetch stock data
        data = fetch_stock_data(ticker, period="1mo")

        # Print the first few rows
        print(f"\nPreview of {ticker} data:")
        print(data.head())

        # Save to CSV
        filepath = save_to_csv(data, ticker)
        print(f"Saved to {filepath}")

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")