# Nifty 50 IT Sector Forecasting Pipeline

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/aniketqxp/stock-timeseries-prediction)

A multivariate time-series analytical framework designed to forecast trend continuations in the Nifty 50 IT sector. This system implements a hybrid architecture combining statistical autoregression (SARIMA) with recursive deep learning (GRU) to model non-linear temporal dependencies across the Indian IT infrastructure.

## System Architecture

```mermaid
graph TD
    A[/Data Ingestion/] -->|yfinance| B(Pre-processing Engine)
    B --> C{Feature Engineering}
    C -->|APS| D[Volume-Weighted Metric]
    C -->|Global| E[Macroeconomic Indicators]
    C -->|Technical| F[SMA / RSI / Lags]
    D & E & F --> G([Hybrid Modeling Suite])
    G --> H[SARIMA Baseline]
    G --> I[Multivariate GRU]
    H & I --> J(Benchmarking & Evaluation)
    J --> K[/10-Day Horizon Projection/]

    %% Styling
    style A fill:#2d3436,stroke:#000,color:#fff
    style B fill:#0984e3,stroke:#000,color:#fff
    style C fill:#fdcb6e,stroke:#000,color:#000
    style D fill:#00b894,stroke:#000,color:#fff
    style E fill:#00b894,stroke:#000,color:#fff
    style F fill:#00b894,stroke:#000,color:#fff
    style G fill:#6c5ce7,stroke:#000,color:#fff
    style H fill:#e17055,stroke:#000,color:#fff
    style I fill:#e17055,stroke:#000,color:#fff
    style J fill:#0984e3,stroke:#000,color:#fff
    style K fill:#2d3436,stroke:#000,color:#fff
```

## Core Methodology

The pipeline utilizes the **Aggregate Performance Score (APS)** as its primary target variable. The APS is a volume-weighted composite metric that captures the sector-wide valuation density of major constituents (TCS, Infosys, Wipro, HCL Tech, Tech Mahindra).

### Analytical Features
*   **Macroeconomic Integration**: Correlation mapping with Gold futures, USD-INR exchange rates, and S&P 500 benchmarks.
*   **Trend Oscillators**: Automated calculation of 10/20/50-day Moving Averages and Relative Strength Index (RSI).
*   **Recursive Forecasting**: Implementation of multi-step recursive projection for a 10-day temporal window.

## Local Environment Setup

### Prerequisites
*   Python 3.10 or higher
*   pip / venv

### Installation
```bash
# Clone the repository
git clone https://github.com/aniketqxp/stock-timeseries-prediction.git
cd stock-timeseries-prediction

# Initialize virtual environment
python -m venv venv
source venv/bin/scripts/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start Usage

### Pipeline Execution
Run the primary entry point to initiate the end-to-end ingestion, training, and projection sequence:
```bash
python main.py
```

### Research Exploration
For interactive diagnostics and exploratory data analysis (EDA), refer to the project notebook:
```bash
jupyter notebook notebooks/stock_prediction_project.ipynb
```

## Visual Outputs

![Sector Performance Analysis](assets/sector_analysis.png)
*Figure 1: Component price profiling across Nifty 50 IT constituents.*

![Short-term Trend Projection](assets/forecast_projection.png)
*Figure 2: Observed vs. Projected Horizon (10-Day Recursive Forecast).*
