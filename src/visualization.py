import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(historical_dates, historical_values, forecast_dates, forecast_values, title="Sector Performance Projection", save_path=None):
    """
    Plots historical vs forecasted values with analytical context.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(historical_dates, historical_values, label='Historical APS', marker='o', markersize=4, linewidth=2, color='#3498db')
    plt.plot(forecast_dates, forecast_values, label='Projected Horizon', color='#e74c3c', marker='x', markersize=6, linestyle='--', linewidth=2)
    
    # Analytical Context
    plt.axvline(x=historical_dates[-1], color='#7f8c8d', linestyle='-', alpha=0.5, label='Forecast Boundary')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.xlabel('Temporal Index (Date)')
    plt.ylabel('Aggregate Performance Score (APS)')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"High-resolution plot saved to {save_path}")
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
    Plots individual ticker prices for sector analysis with enhanced formatting.
    """
    plt.figure(figsize=(15, 7))
    for ticker in tickers:
        if ticker in adj_prices.columns:
            plt.plot(adj_prices.index, adj_prices[ticker], label=ticker, alpha=0.7, linewidth=1.5)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Adjusted Close Price (INR)')
    plt.legend(loc='upper left', ncol=2, frameon=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"High-resolution sector analysis saved to {save_path}")
    else:
        plt.show()
