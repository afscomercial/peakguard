
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
from .utils import (
    prepare_hourly_from_kaggle_like,
    generate_synthetic_consumption,
    load_artifacts,
    forecast_next_hours_from_hourly,
)

APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(APP_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gru_energy_forecaster.keras')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'series_minmax_scaler.pkl')

app = FastAPI(title="PeakGuard API")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory synthetic stream state
STATE = {"history": None}

class ForecastQuery(BaseModel):
    # We accept 'steps' for API compatibility, but we will forecast only 1 step
    steps: int = 1
    nowMs: int | None = None


def _now_from_ms(now_ms: int | None) -> pd.Timestamp:
    if now_ms is not None:
        # Interpret as browser local epoch ms
        return pd.Timestamp.fromtimestamp(now_ms / 1000.0)
    # Fallback to server local time
    return pd.Timestamp.now()

@app.on_event("startup")
async def startup_event():
    # Initialize with 24h seed + some buffer (48h) to display
    start = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d %H:00:00')
    synth = generate_synthetic_consumption(start=start, periods=24*2, freq='H', seed=123)
    hourly = prepare_hourly_from_kaggle_like(synth)
    STATE["history"] = hourly

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/synthetic/latest", response_class=JSONResponse)
async def synthetic_latest(nowMs: int | None = None):
    # Extend by 1 hour each call to simulate IoT stream
    hist = STATE["history"]
    # Align to browser local time (hourly)
    now_hour = _now_from_ms(nowMs).floor('H')
    # Extend history up to now_hour
    while hist.index[-1] < now_hour:
        last_time = hist.index[-1]
        next_time = last_time + pd.Timedelta(hours=1)
        recent = hist.iloc[-24:]['consumption']
        ref = pd.Series(recent.values)
        one = generate_synthetic_consumption(start=next_time.strftime('%Y-%m-%d %H:00:00'), periods=1, freq='H', seed=int(next_time.timestamp()), reference_hourly=ref)
        one_hourly = prepare_hourly_from_kaggle_like(one)
        hist = pd.concat([hist, one_hourly])
        STATE["history"] = hist
    # Return last 24 hours for the live list (ending at now_hour)
    if hist.index[-1] > now_hour:
        # Slice up to now_hour if we ran ahead
        hist = hist.loc[:now_hour]
        STATE["history"] = hist
    tail = hist.iloc[-24:]
    return {"timestamps": tail.index.strftime('%Y-%m-%d %H:%M:%S').tolist(), "consumption": tail['consumption'].round(4).tolist()}

@app.post("/api/forecast", response_class=JSONResponse)
async def forecast(query: ForecastQuery):
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
    hist = STATE["history"]
    now_hour = _now_from_ms(query.nowMs).floor('H')
    # Ensure history is extended to now_hour
    while hist.index[-1] < now_hour:
        last_time = hist.index[-1]
        next_time = last_time + pd.Timedelta(hours=1)
        recent = hist.iloc[-24:]['consumption']
        ref = pd.Series(recent.values)
        one = generate_synthetic_consumption(start=next_time.strftime('%Y-%m-%d %H:00:00'), periods=1, freq='H', seed=int(next_time.timestamp()), reference_hourly=ref)
        one_hourly = prepare_hourly_from_kaggle_like(one)
        hist = pd.concat([hist, one_hourly])
        STATE["history"] = hist
    if hist.index[-1] > now_hour:
        hist = hist.loc[:now_hour]
        STATE["history"] = hist
    # Always forecast only next 1 hour; ignore larger requests
    pred_df = forecast_next_hours_from_hourly(hist, model, scaler, window_size=24, steps=1)
    # Return last 24h of history for plotting context
    tail = hist.iloc[-24:]
    return {
        "history": {
            "timestamps": tail.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "consumption": tail['consumption'].round(4).tolist(),
        },
        "forecast": {
            "timestamps": pred_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y_pred": pred_df['y_pred'].round(4).tolist(),
        },
    }
