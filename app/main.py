
import os
import asyncio
import sqlite3
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
    forecast_next_hours_general,
    get_model_window_size,
)
from . import db as dbmod
import json

APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(APP_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
LATEST_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, 'latest')
MODEL_PATH = os.path.join(LATEST_ARTIFACTS_DIR, 'gru_energy_forecaster.keras')
SCALER_PATH = os.path.join(LATEST_ARTIFACTS_DIR, 'series_minmax_scaler.pkl')

app = FastAPI(title="PeakGuard API")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory synthetic stream state
STATE = {"history": None, "bg_task": None}

class ForecastQuery(BaseModel):
    # We accept 'steps' for API compatibility, but we will forecast only 1 step
    steps: int = 1
    nowMs: int | None = None
    tzOffsetMin: int | None = None
    deviceId: int | None = None


class AdminGenerateRequest(BaseModel):
    deviceId: int
    hours: int | None = 72  # generate last N hours up to device-local now

class AdminGenerateRangeRequest(BaseModel):
    deviceId: int
    startUtc: str
    endUtc: str


def _now_from_ms(now_ms: int | None, tz_offset_min: int | None = None) -> pd.Timestamp:
    if now_ms is not None:
        # Build tz-aware UTC timestamp
        utc_ts = pd.Timestamp(now_ms, unit='ms', tz='UTC')
        if tz_offset_min is not None:
            # Browser local time = UTC - offset minutes
            local_ts = utc_ts - pd.Timedelta(minutes=tz_offset_min)
            return local_ts.tz_convert(None)
        # No offset provided: return UTC as naive
        return utc_ts.tz_convert(None)
    # Fallback to server UTC time (naive)
    return pd.Timestamp.utcnow()

@app.on_event("startup")
async def startup_event():
    # DB migrations and one-time bootstrap of devices and synthetic history
    with dbmod.get_conn() as conn:
        dbmod.migrate(conn)
        if dbmod.get_meta(conn, "seeded") != "1":
            devices = [
                (1, "Device A", "America/New_York"),
                (2, "Device B", "Europe/Berlin"),
            ]
            dbmod.upsert_devices(conn, devices)
            # Seed 60 days hourly data per device
            for dev_id, _, tz in devices:
                # Generate in device-local time and convert to UTC for storage
                end_local = dbmod.device_local_now_hour(tz)
                start_local = end_local - pd.Timedelta(days=60)
                start_str = start_local.strftime('%Y-%m-%d %H:00:00')
                periods = int(((end_local - start_local).total_seconds() // 3600) + 1)
                synth = generate_synthetic_consumption(start=start_str, periods=periods, freq='H', seed=dev_id * 1000 + 42)
                hourly = prepare_hourly_from_kaggle_like(synth)
                # Persist as UTC
                rows = []
                for ts_local, cons in hourly['consumption'].items():
                    ts_utc = dbmod.to_utc_from_device_local(pd.Timestamp(ts_local), tz)
                    rows.append((ts_utc.strftime('%Y-%m-%d %H:%M:%S'), float(cons)))
                dbmod.insert_readings(conn, dev_id, rows)
            dbmod.set_meta(conn, "seeded", "1")
    # Keep in-memory history for backward compatibility (default device 1)
    with dbmod.get_conn() as conn:
        end_utc = pd.Timestamp.utcnow().floor('H')
        start_utc = end_utc - pd.Timedelta(hours=24)
        df = dbmod.fetch_readings_range(conn, 1, start_utc, end_utc)
        STATE["history"] = df
    # Launch background generator loop
    STATE["bg_task"] = asyncio.create_task(_generator_loop())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/devices", response_class=JSONResponse)
async def list_devices():
    with dbmod.get_conn() as conn:
        return dbmod.list_devices(conn)

@app.get("/api/models/latest", response_class=JSONResponse)
async def latest_model_metrics():
    with dbmod.get_conn() as conn:
        res = dbmod.get_latest_model_results(conn)
        if not res:
            return {"message": "No trained model available yet"}
        return res

@app.post("/api/admin/generate", response_class=JSONResponse)
async def admin_generate(req: AdminGenerateRequest):
    dev_id = req.deviceId
    hours = req.hours or 72
    if hours < 1:
        return JSONResponse(status_code=400, content={"error": "hours must be >= 1"})
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        now_local = dbmod.device_local_now_hour(dev["timezone"]) 
        start_utc = dbmod.to_utc_from_device_local(now_local - pd.Timedelta(hours=hours - 1), dev["timezone"]) 
        end_utc = dbmod.to_utc_from_device_local(now_local, dev["timezone"]) 
        await _generate_range_for_device(conn, dev_id, dev["timezone"], start_utc, end_utc)
        # Return simple stats
        df = dbmod.fetch_readings_range(conn, dev_id, start_utc, end_utc)
        return {
            "deviceId": dev_id,
            "generated_window_utc": {
                "start": start_utc.strftime('%Y-%m-%d %H:%M:%S'),
                "end": end_utc.strftime('%Y-%m-%d %H:%M:%S'),
            },
            "row_count": int(len(df)),
        }

@app.post("/api/admin/generate-range", response_class=JSONResponse)
async def admin_generate_range(req: AdminGenerateRangeRequest):
    dev_id = req.deviceId
    try:
        start_utc = pd.to_datetime(req.startUtc)
        end_utc = pd.to_datetime(req.endUtc)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid startUtc/endUtc"})
    if end_utc < start_utc:
        return JSONResponse(status_code=400, content={"error": "endUtc must be >= startUtc"})
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        await _generate_range_for_device(conn, dev_id, dev["timezone"], start_utc, end_utc)
        df = dbmod.fetch_readings_range(conn, dev_id, start_utc, end_utc)
        return {
            "deviceId": dev_id,
            "generated_window_utc": {
                "start": start_utc.strftime('%Y-%m-%d %H:%M:%S'),
                "end": end_utc.strftime('%Y-%m-%d %H:%M:%S'),
            },
            "row_count": int(len(df)),
        }

@app.get("/api/admin/forecast-window", response_class=JSONResponse)
async def admin_forecast_window(deviceId: int, fill: bool = False):
    """Inspect the exact window the forecast endpoint requires and optionally backfill it.
    Returns: window size, UTC start/end, row_count, missing_count, first missing timestamps.
    """
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
    win = get_model_window_size(model)
    dev_id = deviceId
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        now_local = dbmod.device_local_now_hour(dev["timezone"]) 
        start_utc = dbmod.to_utc_from_device_local(now_local - pd.Timedelta(hours=win - 1), dev["timezone"]) 
        end_utc = dbmod.to_utc_from_device_local(now_local, dev["timezone"]) 
        if fill:
            await _ensure_window_filled(conn, dev_id, dev["timezone"], start_utc, end_utc)
        df = dbmod.fetch_readings_range(conn, dev_id, start_utc, end_utc)
        expected = pd.date_range(start=start_utc, end=end_utc, freq='H')
        actual = df.index if not df.empty else pd.DatetimeIndex([])
        missing = expected.difference(actual)
        return {
            "deviceId": dev_id,
            "windowHours": int(win),
            "utc_window": {
                "start": start_utc.strftime('%Y-%m-%d %H:%M:%S'),
                "end": end_utc.strftime('%Y-%m-%d %H:%M:%S'),
            },
            "row_count": int(len(df)),
            "missing_count": int(len(missing)),
            "missing_first": [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in missing[:10]],
        }

@app.get("/api/synthetic/latest", response_class=JSONResponse)
async def synthetic_latest(nowMs: int | None = None, tzOffsetMin: int | None = None, deviceId: int | None = None):
    # Device selection defaults to 1
    dev_id = deviceId or 1
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        # Always align to device-local wall clock for consistency by device
        now_device_local = dbmod.device_local_now_hour(dev["timezone"]) 
        start_utc = dbmod.to_utc_from_device_local(now_device_local - pd.Timedelta(hours=23), dev["timezone"]) 
        end_utc = dbmod.to_utc_from_device_local(now_device_local, dev["timezone"]) 

        # Ensure the last 24h window exists in DB (idempotent insert)
        await _generate_range_for_device(conn, dev_id, dev["timezone"], start_utc, end_utc)
        # Fetch the last 24h window and convert to device-local for display
        df = dbmod.fetch_readings_range(conn, dev_id, start_utc, end_utc)
        if df.empty:
            return {"timestamps": [], "consumption": []}
        # Convert UTC index to device-local naive for output formatting
        idx_local = [dbmod.to_device_local_from_utc(ts, dev["timezone"]) for ts in df.index]
        return {
            "timestamps": pd.DatetimeIndex(idx_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "consumption": df['consumption'].round(4).tolist(),
        }

@app.post("/api/forecast", response_class=JSONResponse)
async def forecast(query: ForecastQuery):
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
    win = get_model_window_size(model)
    # Select device (default 1) and construct last N-hour history (N = model window)
    dev_id = query.deviceId or 1
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        # Always align to device-local clock for consistency by device
        now_device_local = dbmod.device_local_now_hour(dev["timezone"]) 
        start_utc = dbmod.to_utc_from_device_local(now_device_local - pd.Timedelta(hours=win - 1), dev["timezone"]) 
        end_utc = dbmod.to_utc_from_device_local(now_device_local, dev["timezone"]) 
        # Proactively fill just the missing stamps within the model window
        await _ensure_window_filled(conn, dev_id, dev["timezone"], start_utc, end_utc)
        hist = dbmod.fetch_readings_range(conn, dev_id, start_utc, end_utc)
        if hist.empty or len(hist) < win:
            return JSONResponse(status_code=400, content={"error": f"Insufficient history for forecasting (need {win})"})
        # Forecast next hour based on UTC-indexed series
        pred_df = forecast_next_hours_general(hist, model, scaler, window_size=win, steps=1, device_id=dev_id, device_tz=dev["timezone"]) 
        # Convert outputs to device-local for presentation
        tail = hist.iloc[-24:]
        hist_idx_local = [dbmod.to_device_local_from_utc(ts, dev["timezone"]) for ts in tail.index]
        pred_idx_local = [dbmod.to_device_local_from_utc(ts, dev["timezone"]) for ts in pred_df['timestamp']]
    return {
        "history": {
            "timestamps": pd.DatetimeIndex(hist_idx_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "consumption": tail['consumption'].round(4).tolist(),
        },
        "forecast": {
            "timestamps": pd.to_datetime(pred_idx_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y_pred": pred_df['y_pred'].round(4).tolist(),
        },
    }


async def _generate_missing_for_device(conn: sqlite3.Connection | None, device_id: int, device_tz: str) -> None:
    # Ensure we have a connection
    owns_conn = False
    if conn is None:
        owns_conn = True
        ctx = dbmod.get_conn()
        conn = ctx.__enter__()
    try:
        now_device_local = dbmod.device_local_now_hour(device_tz)
        end_utc = dbmod.to_utc_from_device_local(now_device_local, device_tz)
        last_utc = dbmod.get_last_reading_utc(conn, device_id)
        if last_utc is None or last_utc < end_utc:
            gen_start_utc = (last_utc + pd.Timedelta(hours=1)) if last_utc is not None else (end_utc - pd.Timedelta(hours=48))
            periods = int(((end_utc - gen_start_utc).total_seconds() // 3600) + 1)
            # Reference last 24h
            ref_df = dbmod.fetch_readings_range(conn, device_id, end_utc - pd.Timedelta(hours=48), end_utc)
            ref_series = ref_df['consumption'].tail(24) if not ref_df.empty else None
            gen_start_local = dbmod.to_device_local_from_utc(gen_start_utc, device_tz)
            synth = generate_synthetic_consumption(
                start=gen_start_local.strftime('%Y-%m-%d %H:00:00'),
                periods=periods,
                freq='H',
                seed=device_id * 1000 + int(end_utc.timestamp()),
                reference_hourly=(ref_series if ref_series is not None and len(ref_series) > 0 else None),
            )
            hourly = prepare_hourly_from_kaggle_like(synth)
            rows = []
            for ts_local, cons in hourly['consumption'].items():
                ts_utc = dbmod.to_utc_from_device_local(pd.Timestamp(ts_local), device_tz)
                rows.append((ts_utc.strftime('%Y-%m-%d %H:%M:%S'), float(cons)))
            dbmod.insert_readings(conn, device_id, rows)
    finally:
        if owns_conn:
            ctx.__exit__(None, None, None)


def _seconds_until_next_utc_hour() -> float:
    now = pd.Timestamp.utcnow()
    next_hour = (now.floor('H') + pd.Timedelta(hours=1))
    return float((next_hour - now).total_seconds())


async def _generator_loop():
    try:
        while True:
            # Sleep until next UTC hour boundary
            await asyncio.sleep(max(1.0, _seconds_until_next_utc_hour()))
            # At each hour, iterate devices and generate one hour up to device-local now
            with dbmod.get_conn() as conn:
                devices = dbmod.list_devices(conn)
                for d in devices:
                    await _generate_missing_for_device(conn, d['id'], d['timezone'])
    except asyncio.CancelledError:
        # Normal shutdown path
        return


@app.on_event("shutdown")
async def shutdown_event():
    task = STATE.get("bg_task")
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _generate_range_for_device(conn: sqlite3.Connection, device_id: int, device_tz: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> None:
    if end_utc < start_utc:
        return
    periods = int(((end_utc - start_utc).total_seconds() // 3600) + 1)
    # Reference last 24h before start
    ref_df = dbmod.fetch_readings_range(conn, device_id, start_utc - pd.Timedelta(hours=48), start_utc)
    ref_series = ref_df['consumption'].tail(24) if not ref_df.empty else None
    start_local = dbmod.to_device_local_from_utc(start_utc, device_tz)
    synth = generate_synthetic_consumption(
        start=start_local.strftime('%Y-%m-%d %H:00:00'),
        periods=periods,
        freq='H',
        seed=device_id * 1000 + int(end_utc.timestamp()),
        reference_hourly=(ref_series if ref_series is not None and len(ref_series) > 0 else None),
    )
    hourly = prepare_hourly_from_kaggle_like(synth)
    rows = []
    for ts_local, cons in hourly['consumption'].items():
        ts_utc = dbmod.to_utc_from_device_local(pd.Timestamp(ts_local), device_tz)
        rows.append((ts_utc.strftime('%Y-%m-%d %H:%M:%S'), float(cons)))
    dbmod.insert_readings(conn, device_id, rows)


async def _ensure_window_filled(conn: sqlite3.Connection, device_id: int, device_tz: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> None:
    """Ensure every hourly timestamp in [start_utc, end_utc] exists by generating only missing spans."""
    df = dbmod.fetch_readings_range(conn, device_id, start_utc, end_utc)
    expected = pd.date_range(start=start_utc, end=end_utc, freq='H')
    actual = df.index if not df.empty else pd.DatetimeIndex([])
    missing = expected.difference(actual)
    if len(missing) == 0:
        return
    # Group missing into contiguous ranges
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = None
    for ts in missing:
        if start is None:
            start = prev = ts
            continue
        if ts == prev + pd.Timedelta(hours=1):
            prev = ts
            continue
        ranges.append((start, prev))
        start = prev = ts
    if start is not None:
        ranges.append((start, prev))
    # Generate each missing block
    for s, e in ranges:
        await _generate_range_for_device(conn, device_id, device_tz, s, e)
