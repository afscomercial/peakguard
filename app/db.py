import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple, Dict
import pandas as pd
from zoneinfo import ZoneInfo
import json


DEFAULT_DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "peakguard.db"),
)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


@contextmanager
def get_conn(db_path: Optional[str] = None):
    path = db_path or DEFAULT_DB_PATH
    ensure_parent_dir(path)
    # If DB file is missing or empty, seed from baked snapshot if present
    try:
        need_seed = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if need_seed:
            seed_path = os.environ.get("SEED_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "seed", "peakguard.db"))
            if os.path.exists(seed_path) and os.path.getsize(seed_path) > 0:
                ensure_parent_dir(path)
                import shutil
                shutil.copyfile(seed_path, path)
    except Exception:
        # Non-fatal; proceed to create a new DB
        pass
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.commit()
        conn.close()


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            timezone TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            ts_utc TEXT NOT NULL,
            consumption REAL NOT NULL,
            UNIQUE(device_id, ts_utc),
            FOREIGN KEY(device_id) REFERENCES devices(id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            notes TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_results (
            model_id INTEGER PRIMARY KEY,
            loss_history TEXT NOT NULL,
            rmse_history TEXT NOT NULL,
            test_plot TEXT NOT NULL,
            FOREIGN KEY(model_id) REFERENCES models(id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            ts_utc TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            y_pred REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(device_id, ts_utc, model_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            ts_utc TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            rmse_24h REAL,
            mape_24h REAL,
            baseline_rmse_24h REAL,
            rmse_ratio_24h REAL,
            bias_24h REAL,
            created_at TEXT NOT NULL,
            UNIQUE(device_id, ts_utc, model_id)
        );
        """
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def upsert_devices(conn: sqlite3.Connection, devices: List[Tuple[int, str, str]]) -> None:
    conn.executemany(
        "INSERT INTO devices(id, name, timezone) VALUES(?, ?, ?)\n         ON CONFLICT(id) DO UPDATE SET name=excluded.name, timezone=excluded.timezone",
        devices,
    )


def list_devices(conn: sqlite3.Connection) -> List[Dict]:
    cur = conn.execute("SELECT id, name, timezone FROM devices ORDER BY id")
    return [
        {"id": row[0], "name": row[1], "timezone": row[2]}
        for row in cur.fetchall()
    ]


def get_device(conn: sqlite3.Connection, device_id: int) -> Optional[Dict]:
    cur = conn.execute("SELECT id, name, timezone FROM devices WHERE id=?", (device_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "timezone": row[2]}


def get_last_reading_utc(conn: sqlite3.Connection, device_id: int) -> Optional[pd.Timestamp]:
    cur = conn.execute(
        "SELECT ts_utc FROM readings WHERE device_id=? ORDER BY ts_utc DESC LIMIT 1",
        (device_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    ts_utc_aware = pd.to_datetime(row[0], utc=True)
    return ts_utc_aware.tz_convert(None)


def insert_readings(conn: sqlite3.Connection, device_id: int, rows: Iterable[Tuple[str, float]]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO readings(device_id, ts_utc, consumption) VALUES(?, ?, ?)",
        ((device_id, ts_utc, cons) for ts_utc, cons in rows),
    )


def insert_model(conn: sqlite3.Connection, artifact_dir: str, created_at: Optional[str] = None, notes: Optional[str] = None) -> int:
    created = created_at or pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "INSERT INTO models(created_at, artifact_dir, notes) VALUES(?, ?, ?)",
        (created, artifact_dir, notes),
    )
    return int(cur.lastrowid)


def save_model_results(conn: sqlite3.Connection, model_id: int, loss_history: List[float], rmse_history: List[float], test_plot: Dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO model_results(model_id, loss_history, rmse_history, test_plot) VALUES(?, ?, ?, ?)",
        (model_id, json.dumps(loss_history), json.dumps(rmse_history), json.dumps(test_plot)),
    )


def get_latest_model_results(conn: sqlite3.Connection) -> Optional[Dict]:
    cur = conn.execute(
        "SELECT m.id, m.created_at, m.artifact_dir, r.loss_history, r.rmse_history, r.test_plot\n         FROM models m JOIN model_results r ON r.model_id = m.id\n         ORDER BY m.id DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "model_id": row[0],
        "created_at": row[1],
        "artifact_dir": row[2],
        "loss_history": json.loads(row[3]),
        "rmse_history": json.loads(row[4]),
        "test_plot": json.loads(row[5]),
    }


def list_model_ids(conn: sqlite3.Connection) -> List[int]:
    cur = conn.execute("SELECT id FROM models ORDER BY id ASC")
    return [int(r[0]) for r in cur.fetchall()]


def upsert_model_with_id(
    conn: sqlite3.Connection,
    model_id: int,
    created_at: str,
    artifact_dir: str,
    notes: Optional[str],
) -> None:
    conn.execute(
        "INSERT INTO models(id, created_at, artifact_dir, notes) VALUES(?, ?, ?, ?)\n         ON CONFLICT(id) DO UPDATE SET created_at=excluded.created_at, artifact_dir=excluded.artifact_dir, notes=excluded.notes",
        (model_id, created_at, artifact_dir, notes),
    )


def get_model_with_results(conn: sqlite3.Connection, model_id: int) -> Optional[Dict]:
    cur = conn.execute(
        "SELECT id, created_at, artifact_dir, notes FROM models WHERE id=?",
        (model_id,),
    )
    m = cur.fetchone()
    if not m:
        return None
    cur = conn.execute(
        "SELECT loss_history, rmse_history, test_plot FROM model_results WHERE model_id=?",
        (model_id,),
    )
    r = cur.fetchone()
    loss = json.loads(r[0]) if r and r[0] else {"train": [], "val": []}
    rmse = json.loads(r[1]) if r and r[1] else []
    plot = json.loads(r[2]) if r and r[2] else {"y_true": [], "y_pred": []}
    return {
        "id": int(m[0]),
        "created_at": m[1],
        "artifact_dir": m[2],
        "notes": m[3],
        "loss_history": loss,
        "rmse_history": rmse,
        "test_plot": plot,
    }


def insert_prediction(conn: sqlite3.Connection, device_id: int, ts_utc: pd.Timestamp, model_id: int, y_pred: float) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO predictions(device_id, ts_utc, model_id, y_pred, created_at) VALUES(?, ?, ?, ?, ?)",
        (device_id, ts_utc.strftime("%Y-%m-%d %H:%M:%S"), model_id, float(y_pred), pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
    )


def fetch_latest_prediction(conn: sqlite3.Connection, device_id: int) -> Optional[Dict]:
    cur = conn.execute(
        "SELECT ts_utc, model_id, y_pred FROM predictions WHERE device_id=? ORDER BY ts_utc DESC LIMIT 1",
        (device_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    return {"ts_utc": r[0], "model_id": int(r[1]), "y_pred": float(r[2])}


def fetch_predictions_range(conn: sqlite3.Connection, device_id: int, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    cur = conn.execute(
        """
        SELECT ts_utc, model_id, y_pred
        FROM predictions
        WHERE device_id=? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc ASC
        """,
        (device_id, start_utc.strftime("%Y-%m-%d %H:%M:%S"), end_utc.strftime("%Y-%m-%d %H:%M:%S")),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"), columns=["model_id", "y_pred"])
    idx = pd.to_datetime([r[0] for r in rows])
    df = pd.DataFrame({"timestamp": idx, "model_id": [int(r[1]) for r in rows], "y_pred": [float(r[2]) for r in rows]}).set_index("timestamp")
    return df


def upsert_health_metrics(
    conn: sqlite3.Connection,
    device_id: int,
    ts_utc: pd.Timestamp,
    model_id: int,
    rmse_24h: float,
    mape_24h: float,
    baseline_rmse_24h: float,
    rmse_ratio_24h: float,
    bias_24h: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO health_metrics(device_id, ts_utc, model_id, rmse_24h, mape_24h, baseline_rmse_24h, rmse_ratio_24h, bias_24h, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (device_id, ts_utc.strftime("%Y-%m-%d %H:%M:%S"), model_id, rmse_24h, mape_24h, baseline_rmse_24h, rmse_ratio_24h, bias_24h, pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
    )


def fetch_latest_health(conn: sqlite3.Connection, device_id: int) -> Optional[Dict]:
    cur = conn.execute(
        "SELECT ts_utc, model_id, rmse_24h, mape_24h, baseline_rmse_24h, rmse_ratio_24h, bias_24h FROM health_metrics WHERE device_id=? ORDER BY ts_utc DESC LIMIT 1",
        (device_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    return {
        "ts_utc": r[0],
        "model_id": int(r[1]),
        "rmse_24h": float(r[2]) if r[2] is not None else None,
        "mape_24h": float(r[3]) if r[3] is not None else None,
        "baseline_rmse_24h": float(r[4]) if r[4] is not None else None,
        "rmse_ratio_24h": float(r[5]) if r[5] is not None else None,
        "bias_24h": float(r[6]) if r[6] is not None else None,
    }


def fetch_readings_range(
    conn: sqlite3.Connection,
    device_id: int,
    start_utc_inclusive: pd.Timestamp,
    end_utc_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    cur = conn.execute(
        """
        SELECT ts_utc, consumption
        FROM readings
        WHERE device_id=? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc ASC
        """,
        (
            device_id,
            start_utc_inclusive.strftime("%Y-%m-%d %H:%M:%S"),
            end_utc_inclusive.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"), columns=["consumption"]).astype(float)
    idx = pd.to_datetime([r[0] for r in rows])
    df = pd.DataFrame({"timestamp": idx, "consumption": [r[1] for r in rows]}).set_index("timestamp")
    return df


def device_local_now_hour(device_tz: str) -> pd.Timestamp:
    tz = ZoneInfo(device_tz)
    # Avoid tz_localize; construct as UTC-aware directly
    return pd.Timestamp.now(tz="UTC").tz_convert(tz).floor("H").tz_convert(None)


def to_utc_from_device_local(ts_local: pd.Timestamp, device_tz: str) -> pd.Timestamp:
    tz = ZoneInfo(device_tz)
    ts_local = pd.Timestamp(ts_local)
    if getattr(ts_local, 'tz', None) is None:
        ts_z = ts_local.tz_localize(tz)
    else:
        ts_z = ts_local.tz_convert(tz)
    return ts_z.tz_convert("UTC").tz_convert(None)


def to_device_local_from_utc(ts_utc: pd.Timestamp, device_tz: str) -> pd.Timestamp:
    tz = ZoneInfo(device_tz)
    ts_utc_aware = pd.to_datetime(ts_utc, utc=True)
    return ts_utc_aware.tz_convert(tz).tz_convert(None)


