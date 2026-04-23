import pandas as pd
import numpy as np
import yfinance as yf
from src.preprocessing import preprocess_all, create_sequences
from src.models import train_sarima, forecast_sarima, train_gru, predict_future_gru, evaluate_benchmarks
from src.visualization import plot_forecast
from sklearn.model_selection import train_test_split

def run_pipeline():
    # 1. Configuration
    tickers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
    start_date = "2008-01-01"
    end_date = "2024-12-12"
    forecast_steps = 10
    time_steps = 60
    
    print(f"Starting Phase 3 Pipeline: Multivariate Logic Upgrade...")
    
    # 2. Data Ingestion
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    global_data = yf.download(['GC=F', 'INR=X', '^GSPC'], start=start_date, end=end_date)
    
    # 3. Analytical Preprocessing
    # This includes APS calculation, Global Indicators, SMAs, RSI, and Lags
    scaled_df, scaler = preprocess_all(stock_data, global_data)
    print(f"Features Integrated: {len(scaled_df.columns)}")
    print(f"Feature List: {scaled_df.columns.tolist()}")
    
    # 4. Model Training Preparation
    X, y = create_sequences(scaled_df.values, time_steps, target_col_idx=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. SARIMA Baseline (Univariate on APS)
    # We evaluate it on the same test set period
    aps_train = scaled_df['APS'].iloc[:len(X_train)+time_steps]
    sarima_model = train_sarima(aps_train)
    
    # 6. GRU Multivariate Model
    gru_model, history = train_gru(X_train, y_train, X_test, y_test)
    
    # 7. Comparative Benchmarking
    print("\nEvaluating Model Performance...")
    # GRU evaluation
    gru_test_preds = gru_model.predict(X_test, verbose=0)
    # SARIMA evaluation (predict length of test set)
    sarima_test_preds = sarima_model.predict(n_periods=len(y_test))
    
    benchmark_results = evaluate_benchmarks(sarima_test_preds, gru_test_preds, y_test)
    print("\nBenchmark Suite Results (on Test Set):")
    print(benchmark_results)
    
    # 8. Multivariate Future Forecasting (10 Steps)
    last_sequence = scaled_df.values[-time_steps:]
    gru_forecast_scaled = predict_future_gru(gru_model, last_sequence, steps=forecast_steps)
    
    # Inverse scaling for APS (first column)
    dummy = np.zeros((forecast_steps, len(scaled_df.columns)))
    dummy[:, 0] = gru_forecast_scaled
    forecast_aps = scaler.inverse_transform(dummy)[:, 0]
    
    # 9. Visualization
    last_date = scaled_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
    
    # Inverse scale historical APS for plotting
    hist_dummy = np.zeros((60, len(scaled_df.columns)))
    hist_dummy[:, 0] = scaled_df['APS'].values[-60:]
    hist_aps = scaler.inverse_transform(hist_dummy)[:, 0]
    
    plot_forecast(
        scaled_df.index[-60:], 
        hist_aps, 
        forecast_dates, 
        forecast_aps, 
        title="Nifty 50 IT Sector Trend Projection"
    )
    
    print("\nForecasted APS values for the next 10 days:")
    print(forecast_aps)

if __name__ == "__main__":
    run_pipeline()
