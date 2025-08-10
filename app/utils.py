
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from tensorflow import keras


def prepare_hourly_from_kaggle_like(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'TxnDate', 'TxnTime', 'Consumption'}.issubset(df.columns):
        ts = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'], dayfirst=True, errors='coerce')
        vals = pd.to_numeric(df['Consumption'], errors='coerce')
        tmp = pd.DataFrame({'timestamp': ts, 'consumption': vals}).dropna()
    elif {'timestamp', 'consumption'}.issubset(df.columns):
        tmp = df[['timestamp', 'consumption']].dropna()
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], errors='coerce')
        tmp = tmp.dropna()
    else:
        raise ValueError('Invalid input schema')
    hourly = tmp.set_index('timestamp').sort_index().resample('1H').mean(numeric_only=True)
    hourly['consumption'] = hourly['consumption'].interpolate(method='time').ffill().bfill()
    return hourly


def create_sequences(values: np.ndarray, window_size: int = 24, horizon: int = 1):
    X, y = [], []
    for i in range(len(values) - window_size - horizon + 1):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size:i+window_size+horizon])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


def generate_synthetic_consumption(start: str, periods: int, freq: str = 'H', seed: int = 42,
                                    base_level: float | None = None,
                                    daily_amplitude: float | None = None,
                                    weekly_amplitude: float | None = None,
                                    noise_std: float | None = None,
                                    spike_probability: float = 0.01,
                                    spike_scale: float = 3.0,
                                    reference_hourly: Optional[pd.Series] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if reference_hourly is not None and len(reference_hourly) > 0:
        ref_mean = float(np.nanmean(reference_hourly.values))
        ref_std = float(np.nanstd(reference_hourly.values))
        base_level = ref_mean if base_level is None else base_level
        daily_amplitude = (0.15 * ref_mean) if daily_amplitude is None else daily_amplitude
        weekly_amplitude = (0.08 * ref_mean) if weekly_amplitude is None else weekly_amplitude
        noise_std = (0.5 * ref_std) if noise_std is None else noise_std
    else:
        base_level = 1.0 if base_level is None else base_level
        daily_amplitude = 0.15 if daily_amplitude is None else daily_amplitude
        weekly_amplitude = 0.08 if weekly_amplitude is None else weekly_amplitude
        noise_std = 0.1 if noise_std is None else noise_std
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods)
    daily = np.sin(2 * np.pi * t / 24.0)
    weekly = np.sin(2 * np.pi * t / (24.0 * 7.0))
    eps = rng.normal(0.0, noise_std, size=periods)
    ar = np.zeros(periods)
    phi = 0.6
    for i in range(1, periods):
        ar[i] = phi * ar[i - 1] + eps[i]
    spikes = (rng.random(periods) < spike_probability).astype(float)
    spikes *= rng.lognormal(mean=0.0, sigma=0.5, size=periods) * spike_scale
    values = base_level + daily_amplitude * daily + weekly_amplitude * weekly + ar + spikes
    values = np.clip(values, a_min=0.0, a_max=None)
    out = pd.DataFrame({'timestamp': idx, 'consumption': values})
    out['TxnDate'] = out['timestamp'].dt.strftime('%d %b %Y')
    out['TxnTime'] = out['timestamp'].dt.strftime('%H:%M:%S')
    out['Consumption'] = out['consumption'].astype(float)
    return out[['TxnDate', 'TxnTime', 'Consumption']]


def _build_default_model(window_size: int = 24, horizon: int = 1) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        keras.layers.GRU(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(horizon),
    ])
    return model


def load_artifacts(model_path: str, scaler_path: str):
    """Load model and scaler.
    If full-model load fails due to Keras format mismatch, try loading weights
    from '<model_stem>.weights.h5' into a default architecture.
    """
    scaler = joblib.load(scaler_path)
    model = None
    load_error: Exception | None = None
    try:
        # compile=False to avoid missing custom objects during deserialization
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        load_error = e

    if model is None:
        # Try weights fallback
        weights_path = os.path.splitext(model_path)[0] + ".weights.h5"
        if os.path.exists(weights_path):
            model = _build_default_model(window_size=24, horizon=1)
            try:
                model.load_weights(weights_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load weights from {weights_path}: {e}\nOriginal model load error: {load_error}")
        else:
            raise RuntimeError(
                "Failed to load model: " + str(load_error) +
                f"\nProvide matching-format model or add weights at {weights_path} (save via model.save_weights)."
            )

    return model, scaler


def forecast_next_hours_from_hourly(hourly: pd.DataFrame, model, scaler, window_size: int, steps: int):
    series = hourly['consumption'].astype(float).values
    series_scaled = scaler.transform(series.reshape(-1, 1)).reshape(-1)
    if len(series_scaled) < window_size:
        raise ValueError(f"Need at least {window_size} points")
    window = series_scaled[-window_size:].astype(float).copy()
    preds_scaled = []
    for _ in range(steps):
        x = window.reshape(1, window_size, 1)
        yhat = model.predict(x, verbose=0).ravel()[0]
        preds_scaled.append(yhat)
        window[:-1] = window[1:]
        window[-1] = yhat
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).ravel()
    start_time = hourly.index[-1] + pd.Timedelta(hours=1)
    future_index = pd.date_range(start=start_time, periods=steps, freq='H')
    return pd.DataFrame({'timestamp': future_index, 'y_pred': preds})
