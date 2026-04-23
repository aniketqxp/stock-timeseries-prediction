import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_aps(stock_data):
    """Calculates the volume-weighted Aggregate Performance Score (APS)."""
    # 1. Volume Imputation (Forward fill zero volumes)
    if 'Volume' in stock_data.columns:
        stock_data['Volume'] = stock_data['Volume'].replace(0, np.nan).ffill()
        
    price_col = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
    adj_close = stock_data[price_col]
    volume = stock_data['Volume']
    aps = (adj_close * volume).sum(axis=1) / volume.sum(axis=1)
    return pd.DataFrame({'APS': aps})

def add_technical_indicators(df, column='APS'):
    """Adds SMA and RSI indicators."""
    # Simple Moving Averages
    df['SMA_10'] = df[column].rolling(window=10).mean()
    df['SMA_20'] = df[column].rolling(window=20).mean()
    df['SMA_50'] = df[column].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_lags(df, column='APS', lags=5):
    """Adds lag features for the target column."""
    for i in range(1, lags + 1):
        df[f'{column}_lag_{i}'] = df[column].shift(i)
    return df

def preprocess_all(stock_data, global_data):
    """Full preprocessing pipeline with technical indicators and lags."""
    # 1. APS Calculation
    dataset = calculate_aps(stock_data)
    
    # 2. Add Global Metrics
    g_price_col = 'Adj Close' if 'Adj Close' in global_data.columns else 'Close'
    gold = global_data[g_price_col]['GC=F'].to_frame(name='Gold')
    usd_inr = global_data[g_price_col]['INR=X'].to_frame(name='USD_INR')
    sp500 = global_data[g_price_col]['^GSPC'].to_frame(name='SP500')
    
    dataset = dataset.join([gold, usd_inr, sp500], how='inner')
    dataset['SP500_INR'] = dataset['SP500'] * dataset['USD_INR']
    dataset.drop(columns=['SP500'], inplace=True)
    
    # 3. Add Technical Indicators & Lags
    dataset = add_technical_indicators(dataset)
    dataset = add_lags(dataset)
    
    # 4. Final cleaning (dropping NaNs created by rolling/lags)
    dataset.ffill().bfill()
    dataset.dropna(inplace=True)
    
    # 5. Scaling
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(dataset)
    scaled_df = pd.DataFrame(scaled_values, columns=dataset.columns, index=dataset.index)
    
    return scaled_df, scaler

def create_sequences(data, time_steps=60, target_col_idx=0):
    """
    Creates sequences for LSTM/GRU training.
    Assumes target_col_idx is the column to predict.
    """
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, :])  # Use all features
        y.append(data[i, target_col_idx])  # Target is APS
    return np.array(X), np.array(y)
