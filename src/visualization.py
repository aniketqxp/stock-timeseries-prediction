import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(historical_dates, historical_values, forecast_dates, forecast_values, title="Stock Prediction Forecast", save_path=None):
    """
    Plots historical vs forecasted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, historical_values, label='Historical', marker='o', markersize=4)
    plt.plot(forecast_dates, forecast_values, label='Forecast', color='orange', marker='o', markersize=4)
    plt.xlabel('Date')
    plt.ylabel('APS')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_training_history(history):
    """
    Plots loss curves from training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_sector_analysis(adj_prices, tickers, title="Sector Price Analysis", save_path=None):
    """
    Plots individual ticker prices for sector analysis.
    """
    plt.figure(figsize=(15, 6))
    for ticker in tickers:
        if ticker in adj_prices.columns:
            plt.plot(adj_prices.index, adj_prices[ticker], label=ticker, alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
