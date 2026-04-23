import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_sarima(data: pd.Series, seasonal_m=5):
    """
    Finds the best SARIMA model using auto_arima and fits it.
    """
    print("Performing stepwise search for best SARIMA parameters...")
    model = pm.auto_arima(data, seasonal=True, m=seasonal_m, 
                          stepwise=True, trace=True, 
                          error_action='ignore', suppress_warnings=True)
    return model

def forecast_sarima(model, steps=10):
    """
    Forecasts future steps using a SARIMA model.
    """
    return model.predict(n_periods=steps)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_benchmarks(sarima_preds, gru_preds, actual_values):
    """
    Compares SARIMA and GRU predictions against actual values.
    """
    # Ensure actual_values is a numpy array for consistent indexing
    actual = np.array(actual_values)
    
    results = {
        'Model': ['SARIMA', 'GRU'],
        'MSE': [
            mean_squared_error(actual, sarima_preds),
            mean_squared_error(actual, gru_preds)
        ],
        'MAE': [
            mean_absolute_error(actual, sarima_preds),
            mean_absolute_error(actual, gru_preds)
        ],
        'R2 Score': [
            r2_score(actual, sarima_preds),
            r2_score(actual, gru_preds)
        ]
    }
    return pd.DataFrame(results).set_index('Model')

def build_gru_model(input_shape):
    """
    Builds and compiles a GRU model.
    """
    model = Sequential([
        GRU(units=64, return_sequences=False, input_shape=input_shape),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_gru(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Trains the GRU model with early stopping.
    """
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history

def predict_future_gru(model, last_sequence, steps=10):
    """
    Predicts future values recursively using the GRU model.
    """
    predicted_values = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Predict the next step
        p = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]), verbose=0)
        predicted_values.append(p[0, 0])
        
        # Shift sequence and update with the prediction
        new_row = current_seq[-1].copy()
        new_row[0] = p[0, 0] # Update the target (APS) in the input row
        
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = new_row
        
    return np.array(predicted_values)
