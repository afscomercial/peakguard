
import os
import asyncio
import sqlite3
import numpy as np
from fastapi import FastAPI, Request
from fastapi import UploadFile, File, Form, HTTPException
import urllib.request
import urllib.error
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
REMOTE_SYNC_URL = os.environ.get('REMOTE_SYNC_URL') or "https://peakguard-production.up.railway.app"
app = FastAPI(title="PeakGuard API")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory synthetic stream state
STATE = {"history": None, "bg_task": None, "daily_task": None}

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
                synth = generate_synthetic_consumption(
                    start=start_str, periods=periods, freq='H', seed=dev_id * 1000 + 42
                )
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
    # Launch daily scheduler
    STATE["daily_task"] = asyncio.create_task(_daily_health_scheduler())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the HTML dashboard.

    - Renders `templates/index.html` which drives all frontend calls.
    - No side-effects; static assets are served from `/static`.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/devices", response_class=JSONResponse)
async def list_devices():
    """List registered devices from SQLite.

    Returns: [{ id, name, timezone }]
    """
    with dbmod.get_conn() as conn:
        return dbmod.list_devices(conn)

@app.get("/api/models/latest", response_class=JSONResponse)
async def latest_model_metrics():
    """Return the latest model training results.

    Pulls the most recent row from `models` joined with `model_results` and returns
    loss history, optional RMSE history, and a test-set plot. If no model has been
    trained yet, returns { "message": "No trained model available yet" }.
    """
    with dbmod.get_conn() as conn:
        res = dbmod.get_latest_model_results(conn)
        if not res:
            return {"message": "No trained model available yet"}
        return res

def _get_latest_model_id(conn: sqlite3.Connection) -> int | None:
    cur = conn.execute("SELECT id FROM models ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return int(row[0]) if row else None

@app.post("/api/admin/generate", response_class=JSONResponse)
async def admin_generate(req: AdminGenerateRequest):
    """Generate synthetic readings for the last N hours for a device.

    - Aligns to the device-local hour. Only missing timestamps are inserted (idempotent).
    - Useful to quickly populate recent data windows.
    Returns: generated UTC window and row_count present after the operation.
    """
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

@app.get("/api/admin/models/sync", response_class=JSONResponse)
async def admin_models_sync():
    """Remote helper: report model IDs present in the server DB.

    Used by the local sync driver to determine which models need to be pushed.
    Returns: { server_model_ids: [ ... ] }
    """
    """Synchronize model_results from local (artifacts + local DB) to remote DB.
    This endpoint is meant to be called on the remote server from the local machine via curl.
    It compares model IDs present in the server DB with those from a provided metrics payload
    and inserts any missing rows.
    Usage (from local):
      curl -X POST <server>/api/admin/models/push --data @metrics.json
    For simplicity per request, this GET will just report what's missing.
    """
    with dbmod.get_conn() as conn:
        server_ids = set(dbmod.list_model_ids(conn))
    return {"server_model_ids": sorted(list(server_ids))}

@app.post("/api/admin/models/push", response_class=JSONResponse)
async def admin_models_push(payload: dict):
    """Remote helper: upsert models and model_results from a JSON payload.

    Expects { models: [ { id, created_at, artifact_dir, notes, loss_history, rmse_history, test_plot } ] }.
    Upserts rows into `models` and `model_results`. Returns inserted/updated IDs.
    """
    """Push missing model rows to remote DB.
    Body JSON format:
    {
      "models": [
        {"id": 1, "created_at": "YYYY-mm-dd HH:MM:SS", "artifact_dir": "artifacts/versions/...", "notes": "...",
         "loss_history": {"train": [...], "val": [...]}, "rmse_history": [...], "test_plot": {"y_true": [...], "y_pred": [...]}}
      ]
    }
    """
    if not isinstance(payload, dict) or "models" not in payload:
        raise HTTPException(status_code=400, detail="Invalid payload; expected { models: [...] }")
    models = payload["models"]
    if not isinstance(models, list):
        raise HTTPException(status_code=400, detail="models must be a list")

    inserted: list[int] = []
    updated: list[int] = []
    with dbmod.get_conn() as conn:
        existing = set(dbmod.list_model_ids(conn))
        for m in models:
            mid = int(m.get("id"))
            created_at = m.get("created_at") or pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            artifact_dir = m.get("artifact_dir") or os.path.join(ARTIFACTS_DIR, 'latest')
            notes = m.get("notes")
            dbmod.upsert_model_with_id(conn, mid, created_at, artifact_dir, notes)
            loss_hist = m.get('loss_history') or {"train": [], "val": []}
            if not isinstance(loss_hist, dict):
                loss_hist = {"train": [], "val": loss_hist}
            rmse_hist = m.get('rmse_history') or []
            test_plot = m.get('test_plot') or {"y_true": [], "y_pred": []}
            dbmod.save_model_results(conn, mid, loss_history=loss_hist, rmse_history=rmse_hist, test_plot=test_plot)
            (updated if mid in existing else inserted).append(mid)

    return {"inserted": inserted, "updated": updated}


@app.get("/api/admin/models/sync-local-to-remote", response_class=JSONResponse)
async def admin_models_sync_local_to_remote():
    """Local driver: push missing model rows to the remote server.

    - Reads all local `models` + `model_results` rows
    - Queries remote `/api/admin/models/sync` to get existing IDs
    - Pushes missing ones to remote `/api/admin/models/push`
    Returns details about what was pushed or skipped.
    """
    """Collect all local models + metrics and push missing rows to a remote server.
    Remote base URL is read from REMOTE_SYNC_URL env var (e.g., https://peakguard-production.up.railway.app).
    """
    if not REMOTE_SYNC_URL:
        raise HTTPException(status_code=400, detail="REMOTE_SYNC_URL is not set")
    # Build local payload
    models_payload = []
    with dbmod.get_conn() as conn:
        cur = conn.execute("SELECT id, created_at, artifact_dir, notes FROM models ORDER BY id ASC")
        rows = cur.fetchall()
        for mid, created_at, artifact_dir, notes in rows:
            cur2 = conn.execute("SELECT loss_history, rmse_history, test_plot FROM model_results WHERE model_id=?", (mid,))
            r = cur2.fetchone()
            loss = json.loads(r[0]) if r and r[0] else {"train": [], "val": []}
            rmse = json.loads(r[1]) if r and r[1] else []
            plot = json.loads(r[2]) if r and r[2] else {"y_true": [], "y_pred": []}
            models_payload.append({
                "id": int(mid),
                "created_at": created_at,
                "artifact_dir": artifact_dir,
                "notes": notes,
                "loss_history": loss,
                "rmse_history": rmse,
                "test_plot": plot,
            })
    payload = {"models": models_payload}
    # Query remote existing IDs
    try:
        with urllib.request.urlopen(f"{REMOTE_SYNC_URL}/api/admin/models/sync", timeout=20) as resp:
            body = resp.read()
            remote_info = json.loads(body.decode('utf-8'))
            remote_ids = set(remote_info.get('server_model_ids', []))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach remote sync endpoint: {e}")
    # Filter to only models not present on remote
    to_push = [m for m in models_payload if int(m['id']) not in remote_ids]
    if not to_push:
        return {"pushed": [], "skipped": sorted(list(remote_ids)), "message": "Remote already up to date"}
    try:
        data = json.dumps({"models": to_push}).encode('utf-8')
        req = urllib.request.Request(
            url=f"{REMOTE_SYNC_URL}/api/admin/models/push",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode('utf-8'))
        return {"pushed_ids": [m['id'] for m in to_push], "remote_result": out}
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
        raise HTTPException(status_code=502, detail=f"Remote push failed: {e} {detail}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Remote push error: {e}")

@app.post("/api/admin/generate-range", response_class=JSONResponse)
async def admin_generate_range(req: AdminGenerateRangeRequest):
    """Generate synthetic readings for an explicit UTC range for a device.

    Validates `startUtc` <= `endUtc`. Inserts missing hours only. Returns the
    generated window and the resulting row_count.
    """
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
    """Report the model's required forecast window and optionally backfill it.

    - Reads the deployed model's window size (e.g., 48 hours)
    - Computes the [start,end] UTC window aligned to the device-local hour
    - If fill=true, generates any missing readings in that window
    Returns window metadata, row_count, and any missing timestamps.
    """
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
    """Return the last 24 hours of readings for a device.

    - Aligns to the device-local hour and ensures the full 24h window exists (idempotent)
    - Returns timestamps converted to device-local for display
    """
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
    """Return last-24h history and the next-hour forecast for a device.

    - Reads next-hour prediction from the `predictions` table when available
    - If missing, computes one forecast using the deployed model, stores it, and returns it
    - Always returns the last 24h history from `readings` for charting
    """
    # Serve from stored predictions; fallback to on-the-fly if missing
    dev_id = query.deviceId or 1
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, dev_id)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        now_device_local = dbmod.device_local_now_hour(dev["timezone"]) 
        # Last 24h history for chart
        start_hist_utc = dbmod.to_utc_from_device_local(now_device_local - pd.Timedelta(hours=23), dev["timezone"]) 
        end_hist_utc = dbmod.to_utc_from_device_local(now_device_local, dev["timezone"]) 
        await _ensure_window_filled(conn, dev_id, dev["timezone"], start_hist_utc, end_hist_utc)
        hist = dbmod.fetch_readings_range(conn, dev_id, start_hist_utc, end_hist_utc)
        # Pull stored prediction for next hour if present; otherwise compute once
        next_utc = end_hist_utc + pd.Timedelta(hours=1)
        pred_row = dbmod.fetch_predictions_range(conn, dev_id, next_utc, next_utc)
        if pred_row.empty:
            # Compute one-off and store
            model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
            win = get_model_window_size(model)
            start_win_utc = end_hist_utc - pd.Timedelta(hours=win - 1)
            await _ensure_window_filled(conn, dev_id, dev["timezone"], start_win_utc, end_hist_utc)
            win_hist = dbmod.fetch_readings_range(conn, dev_id, start_win_utc, end_hist_utc)
            if not win_hist.empty and len(win_hist) >= win:
                pred_df = forecast_next_hours_general(win_hist, model, scaler, window_size=win, steps=1, device_id=dev_id, device_tz=dev["timezone"]) 
                y_pred = float(pred_df['y_pred'].iloc[0])
                mid = _get_latest_model_id(conn) or 0
                dbmod.insert_prediction(conn, dev_id, next_utc, mid, y_pred)
                pred_ts_local = dbmod.to_device_local_from_utc(next_utc, dev["timezone"]) 
                pred_idx_local = [pred_ts_local]
                y_pred_list = [y_pred]
            else:
                return JSONResponse(status_code=400, content={"error": "Insufficient history for forecasting"})
        else:
            y_pred = float(pred_row['y_pred'].iloc[0])
            pred_ts_local = dbmod.to_device_local_from_utc(next_utc, dev["timezone"]) 
            pred_idx_local = [pred_ts_local]
            y_pred_list = [y_pred]
        # Convert history for presentation
        tail = hist.iloc[-24:]
        hist_idx_local = [dbmod.to_device_local_from_utc(ts, dev["timezone"]) for ts in tail.index]
        return {
        "history": {
            "timestamps": pd.DatetimeIndex(hist_idx_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "consumption": tail['consumption'].round(4).tolist(),
        },
        "forecast": {
            "timestamps": pd.to_datetime(pred_idx_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y_pred": [round(v, 4) for v in y_pred_list],
        },
        }


@app.get("/api/metrics/compare-24h", response_class=JSONResponse)
async def compare_last_24h(deviceId: int):
    """Compute and return last-24h metrics against stored predictions.

    - Ensures readings/predictions exist for the past 24h
    - Joins actual vs. y_pred and computes RMSE, MAPE, baseline RMSE, RMSE ratio, bias
    - Upserts a `health_metrics` row for the device at the end of the window
    Returns joined series and the computed metrics (with a green/red status).
    """
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, deviceId)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        now_loc = dbmod.device_local_now_hour(dev["timezone"]) 
        end_utc = dbmod.to_utc_from_device_local(now_loc, dev["timezone"]) 
        start_utc = end_utc - pd.Timedelta(hours=23)
        await _ensure_window_filled(conn, deviceId, dev["timezone"], start_utc, end_utc)
        await _backfill_predictions_range(conn, deviceId, dev["timezone"], start_utc, end_utc)
        actual = dbmod.fetch_readings_range(conn, deviceId, start_utc, end_utc)
        preds = dbmod.fetch_predictions_range(conn, deviceId, start_utc, end_utc)
        if actual.empty or preds.empty:
            return {"timestamps": [], "actual": [], "y_pred": [], "metrics": {}}
        joined = actual.join(preds[["y_pred"]], how="inner")
        if joined.empty:
            return {"timestamps": [], "actual": [], "y_pred": [], "metrics": {}}
        err = (joined['y_pred'] - joined['consumption']).astype(float)
        rmse24 = float(np.sqrt(np.mean(np.square(err))))
        mape24 = float(np.mean(np.abs(err) / np.maximum(1e-3, np.abs(joined['consumption']))))
        # baseline
        base_series = dbmod.fetch_readings_range(conn, deviceId, start_utc - pd.Timedelta(hours=24), end_utc - pd.Timedelta(hours=24))
        if not base_series.empty:
            base_join = actual.join(base_series[['consumption']].rename(columns={'consumption':'y_base'}), how='inner').dropna()
            if not base_join.empty:
                base_err = (base_join['y_base'] - base_join['consumption']).astype(float)
                baseline_rmse = float(np.sqrt(np.mean(np.square(base_err))))
            else:
                baseline_rmse = float('nan')
        else:
            baseline_rmse = float('nan')
        rmse_ratio = float(rmse24 / baseline_rmse) if baseline_rmse and not np.isnan(baseline_rmse) and baseline_rmse > 0 else float('nan')
        bias24 = float(np.mean(err))
        mid = _get_latest_model_id(conn) or 0
        dbmod.upsert_health_metrics(conn, deviceId, end_utc, mid, rmse24, mape24, baseline_rmse, rmse_ratio, bias24)
        # Determine status color with simple threshold
        threshold_ratio = 0.8
        status_good = (rmse_ratio <= threshold_ratio) if not np.isnan(rmse_ratio) else False
        ts_local = [dbmod.to_device_local_from_utc(ts, dev["timezone"]) for ts in joined.index]
        return {
            "timestamps": pd.DatetimeIndex(ts_local).strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "actual": joined['consumption'].round(4).tolist(),
            "y_pred": joined['y_pred'].round(4).tolist(),
            "metrics": {
                "rmse_24h": round(rmse24, 4),
                "mape_24h": round(mape24, 4),
                "baseline_rmse_24h": (None if np.isnan(baseline_rmse) else round(baseline_rmse, 4)),
                "rmse_ratio_24h": (None if np.isnan(rmse_ratio) else round(rmse_ratio, 4)),
                "bias_24h": round(bias24, 4),
                "status": ("green" if status_good else "red"),
            }
        }


@app.get("/api/metrics/health", response_class=JSONResponse)
async def latest_health(deviceId: int):
    """Return the latest stored health metrics for a device.

    - Fetches the latest `health_metrics` row for the device
    - Adds a simple status field based on thresholds: rmse_ratio<=0.8 or mape<=0.2
    """
    with dbmod.get_conn() as conn:
        row = dbmod.fetch_latest_health(conn, deviceId)
        if not row:
            return {"message": "No health metrics yet"}
        ratio = row.get('rmse_ratio_24h')
        mape = row.get('mape_24h')
        status = "red"
        if ratio is not None and ratio <= 0.8:
            status = "green"
        elif mape is not None and mape <= 0.2:
            status = "green"
        row['status'] = status
        return row


@app.post("/api/admin/metrics/synthetic-eval", response_class=JSONResponse)
async def admin_synthetic_eval(deviceId: int, hours: int = 24):
    """Backfill readings and predictions for a short window and compute metrics.

    - For `hours` (default 24), ensures readings exist (synthetic) and predictions are backfilled
    - Computes and stores an on-demand health snapshot for the end of that window
    - Useful after deploy to produce initial health without waiting for 24h
    """
    if hours < 1:
        return JSONResponse(status_code=400, content={"error": "hours must be >= 1"})
    with dbmod.get_conn() as conn:
        dev = dbmod.get_device(conn, deviceId)
        if not dev:
            return JSONResponse(status_code=404, content={"error": "Device not found"})
        tz = dev['timezone']
        now_loc = dbmod.device_local_now_hour(tz)
        end_utc = dbmod.to_utc_from_device_local(now_loc, tz)
        start_utc = end_utc - pd.Timedelta(hours=hours - 1)
        await _ensure_window_filled(conn, deviceId, tz, start_utc, end_utc)
        await _backfill_predictions_range(conn, deviceId, tz, start_utc, end_utc)
        actual = dbmod.fetch_readings_range(conn, deviceId, start_utc, end_utc)
        preds = dbmod.fetch_predictions_range(conn, deviceId, start_utc, end_utc)
        if actual.empty or preds.empty:
            return {"message": "Insufficient data for evaluation"}
        joined = actual.join(preds[['y_pred']], how='inner')
        if joined.empty:
            return {"message": "No overlap for evaluation"}
        err = (joined['y_pred'] - joined['consumption']).astype(float)
        rmse = float(np.sqrt(np.mean(np.square(err))))
        mape = float(np.mean(np.abs(err) / np.maximum(1e-3, np.abs(joined['consumption']))))
        base_series = dbmod.fetch_readings_range(conn, deviceId, start_utc - pd.Timedelta(hours=24), end_utc - pd.Timedelta(hours=24))
        if not base_series.empty:
            base_join = actual.join(base_series[['consumption']].rename(columns={'consumption':'y_base'}), how='inner').dropna()
            if not base_join.empty:
                base_err = (base_join['y_base'] - base_join['consumption']).astype(float)
                base_rmse = float(np.sqrt(np.mean(np.square(base_err))))
            else:
                base_rmse = float('nan')
        else:
            base_rmse = float('nan')
        ratio = float(rmse / base_rmse) if base_rmse and not np.isnan(base_rmse) and base_rmse > 0 else float('nan')
        bias = float(np.mean(err))
        mid = _get_latest_model_id(conn) or 0
        dbmod.upsert_health_metrics(conn, deviceId, end_utc, mid, rmse, mape, base_rmse, ratio, bias)
    return {"deviceId": deviceId, "window_hours": hours, "stored": True}


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


def _seconds_until_next_utc_day() -> float:
    now = pd.Timestamp.utcnow()
    next_day = (now.floor('D') + pd.Timedelta(days=1))
    return float((next_day - now).total_seconds())


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
                    # Produce next-hour forecast and store
                    try:
                        dev_id = d['id']
                        tz = d['timezone']
                        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
                        win = get_model_window_size(model)
                        now_loc = dbmod.device_local_now_hour(tz)
                        end_utc = dbmod.to_utc_from_device_local(now_loc, tz)
                        start_win_utc = end_utc - pd.Timedelta(hours=win - 1)
                        await _ensure_window_filled(conn, dev_id, tz, start_win_utc, end_utc)
                        win_hist = dbmod.fetch_readings_range(conn, dev_id, start_win_utc, end_utc)
                        if not win_hist.empty and len(win_hist) >= win:
                            pred_df = forecast_next_hours_general(win_hist, model, scaler, window_size=win, steps=1, device_id=dev_id, device_tz=tz)
                            y_pred = float(pred_df['y_pred'].iloc[0])
                            next_utc = end_utc + pd.Timedelta(hours=1)
                            mid = _get_latest_model_id(conn) or 0
                            dbmod.insert_prediction(conn, dev_id, next_utc, mid, y_pred)
                            # Compute health metrics for last 24h
                            start24 = end_utc - pd.Timedelta(hours=23)
                            actual24 = dbmod.fetch_readings_range(conn, dev_id, start24, end_utc)
                            preds24 = dbmod.fetch_predictions_range(conn, dev_id, start24, end_utc)
                            if not actual24.empty and not preds24.empty:
                                joined = actual24.join(preds24[['y_pred']], how='inner')
                                if len(joined) >= 12:
                                    err = (joined['y_pred'] - joined['consumption']).astype(float)
                                    rmse24 = float(np.sqrt(np.mean(np.square(err))))
                                    mape24 = float(np.mean(np.abs(err) / np.maximum(1e-3, np.abs(joined['consumption']))))
                                    # Baseline: previous-day same-hour if available
                                    baseline_series = dbmod.fetch_readings_range(conn, dev_id, start24 - pd.Timedelta(hours=24), end_utc - pd.Timedelta(hours=24))
                                    if not baseline_series.empty:
                                        base_join = actual24.join(baseline_series[['consumption']].rename(columns={'consumption':'y_base'}), how='inner', rsuffix='_base')
                                        base_join = base_join.dropna()
                                        if not base_join.empty:
                                            base_err = (base_join['y_base'] - base_join['consumption']).astype(float)
                                            baseline_rmse = float(np.sqrt(np.mean(np.square(base_err))))
                                        else:
                                            baseline_rmse = float('nan')
                                    else:
                                        baseline_rmse = float('nan')
                                    rmse_ratio = float(rmse24 / baseline_rmse) if baseline_rmse and not np.isnan(baseline_rmse) and baseline_rmse > 0 else float('nan')
                                    bias24 = float(np.mean(err))
                                    dbmod.upsert_health_metrics(conn, dev_id, end_utc, mid, rmse24, mape24, baseline_rmse, rmse_ratio, bias24)
                    except Exception:
                        # Best-effort; don't crash scheduler
                        pass
    except asyncio.CancelledError:
        # Normal shutdown path
        return


async def _daily_health_scheduler():
    try:
        while True:
            await asyncio.sleep(max(1.0, _seconds_until_next_utc_day()))
            with dbmod.get_conn() as conn:
                devices = dbmod.list_devices(conn)
                for d in devices:
                    try:
                        dev_id = d['id']
                        tz = d['timezone']
                        now_loc = dbmod.device_local_now_hour(tz)
                        end_utc = dbmod.to_utc_from_device_local(now_loc, tz)
                        start24 = end_utc - pd.Timedelta(hours=23)
                        await _ensure_window_filled(conn, dev_id, tz, start24, end_utc)
                        await _backfill_predictions_range(conn, dev_id, tz, start24, end_utc)
                        actual24 = dbmod.fetch_readings_range(conn, dev_id, start24, end_utc)
                        preds24 = dbmod.fetch_predictions_range(conn, dev_id, start24, end_utc)
                        if not actual24.empty and not preds24.empty:
                            joined = actual24.join(preds24[['y_pred']], how='inner')
                            if len(joined) >= 12:
                                err = (joined['y_pred'] - joined['consumption']).astype(float)
                                rmse24 = float(np.sqrt(np.mean(np.square(err))))
                                mape24 = float(np.mean(np.abs(err) / np.maximum(1e-3, np.abs(joined['consumption']))))
                                baseline_series = dbmod.fetch_readings_range(conn, dev_id, start24 - pd.Timedelta(hours=24), end_utc - pd.Timedelta(hours=24))
                                if not baseline_series.empty:
                                    base_join = actual24.join(baseline_series[['consumption']].rename(columns={'consumption':'y_base'}), how='inner', rsuffix='_base').dropna()
                                    if not base_join.empty:
                                        base_err = (base_join['y_base'] - base_join['consumption']).astype(float)
                                        baseline_rmse = float(np.sqrt(np.mean(np.square(base_err))))
                                    else:
                                        baseline_rmse = float('nan')
                                else:
                                    baseline_rmse = float('nan')
                                rmse_ratio = float(rmse24 / baseline_rmse) if baseline_rmse and not np.isnan(baseline_rmse) and baseline_rmse > 0 else float('nan')
                                bias24 = float(np.mean(err))
                                mid = _get_latest_model_id(conn) or 0
                                dbmod.upsert_health_metrics(conn, dev_id, end_utc, mid, rmse24, mape24, baseline_rmse, rmse_ratio, bias24)
                    except Exception:
                        pass
    except asyncio.CancelledError:
        return


async def _backfill_predictions_range(conn: sqlite3.Connection, device_id: int, device_tz: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> None:
    try:
        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
        win = get_model_window_size(model)
    except Exception:
        return
    hours = pd.date_range(start=start_utc, end=end_utc, freq='H')
    for h in hours:
        exist = dbmod.fetch_predictions_range(conn, device_id, h, h)
        if not exist.empty:
            continue
        window_end = h - pd.Timedelta(hours=1)
        window_start = window_end - pd.Timedelta(hours=win - 1)
        if window_end < window_start:
            continue
        await _ensure_window_filled(conn, device_id, device_tz, window_start, window_end)
        win_hist = dbmod.fetch_readings_range(conn, device_id, window_start, window_end)
        if win_hist.empty or len(win_hist) < win:
            continue
        try:
            pred_df = forecast_next_hours_general(win_hist, model, scaler, window_size=win, steps=1, device_id=device_id, device_tz=device_tz)
            y_pred = float(pred_df['y_pred'].iloc[0])
            mid = _get_latest_model_id(conn) or 0
            dbmod.insert_prediction(conn, device_id, h, mid, y_pred)
        except Exception:
            continue


@app.on_event("shutdown")
async def shutdown_event():
    task = STATE.get("bg_task")
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    daily = STATE.get("daily_task")
    if daily is not None:
        daily.cancel()
        try:
            await daily
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
