import yfinance as yf
import pandas as pd
import os
from typing import List

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a list of tickers.
    """
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def fetch_global_metrics(start_date: str, end_date: str) -> dict:
    """
    Fetches global metrics: Gold, USD-INR, S&P 500.
    """
    metrics = {
        'Gold': 'GC=F',
        'USD_INR': 'INR=X',
        'SP500': '^GSPC'
    }
    
    results = {}
    for name, ticker in metrics.items():
        print(f"Fetching {name} data ({ticker})...")
        results[name] = yf.download(ticker, start=start_date, end=end_date)
        
    return results

if __name__ == "__main__":
    # Test fetch
    tickers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
    start = "2008-01-01"
    end = "2024-12-12"
    df = fetch_stock_data(tickers, start, end)
    print(df.head())
