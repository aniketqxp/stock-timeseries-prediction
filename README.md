# Stock Time-Series Prediction Pipeline

A professional, production-grade pipeline for predicting stock performance scores using Nifty 50 IT sector data.

## Project Structure

```text
.
├── main.py                 # Main entry point to run the pipeline
├── src/                    # Core logic modules
│   ├── data_ingestion.py   # Yahoo Finance data fetching
│   ├── preprocessing.py    # Cleaning, feature engineering, scaling
│   ├── models.py           # SARIMA and GRU model implementation
│   └── visualization.py    # Plotting and charting logic
├── notebooks/              # Iterative analysis and exploration
│   └── stock_prediction_project.ipynb
├── scripts/                # Utility scripts (deployment, batch jobs)
├── requirements.txt        # Pinned dependencies
└── pyproject.toml         # Modern project configuration
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```

## Key Features

- **Multi-Source Ingestion**: Combines stock ticker data with global economic indicators (Gold, Currency, S&P 500).
- **Aggregate Performance Score (APS)**: A custom volume-weighted metric for trend analysis.
- **Hybrid Modeling**: 
  - **SARIMA**: Statistical time-series forecasting with seasonal awareness.
  - **GRU**: Deep Learning Recurrent Neural Networks for complex pattern recognition.
- **Automated Preprocessing**: Robust alignment and cleaning of non-contiguous time series.
