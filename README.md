## PeakGuard API

FastAPI service that simulates IoT energy consumption per device and serves forecasts trained locally. There are two training options:
- TensorFlow/Keras GRU (global model with device features) — `experiments/TrainingPipeline.ipynb`
- H2O-3 AutoML (tabular regression with lags) — `experiments/H2OTrainingPipeline.ipynb`

### Highlights
- Device-aware synthetic stream stored in SQLite, aligned to each device's timezone
- Background scheduler generates new hourly readings per device (no on-demand generation in endpoints)
- Forecast endpoint returns only the next hour (using the last 24 hours as context)
- Responsive frontend (Plotly.js) showing 24h history and the next-hour forecast, rendered as a continuous segment
- Device selector at the top of the dashboard

### Project layout
```
peakguard_api/
  app/
    __init__.py
    main.py            # FastAPI app and routes
    db.py              # SQLite schema, migrations, queries
    utils.py           # preprocessing, synthetic generator, inference helpers
    artifacts/         # runtime artifacts
      gru_energy_forecaster.keras            # optional (may fail to deserialize across TF/Keras versions)
      gru_energy_forecaster.weights.h5       # preferred fallback (weights-only)
      series_minmax_scaler.pkl               # MinMax scaler for inputs/outputs
  templates/
    index.html         # UI
  static/
    css/styles.css
    js/app.js
  experiments/
    PeakGuard.ipynb             # reference only
    TrainingPipeline.ipynb      # Keras GRU training
    H2OTrainingPipeline.ipynb   # H2O-3 AutoML training
  pyproject.toml
  README.md
```

### Requirements
- macOS Apple Silicon (arm64) supported out-of-the-box
- Python 3.10–3.11
- Poetry (recommended)

### Install (Poetry)
```
cd /Users/andressalguero/Documents/peakguard_api
poetry install --no-interaction --no-ansi --no-root
```

The project targets Apple Silicon with:
- `tensorflow-macos==2.15.0`
- `tensorflow-metal>=1.1.0`
- `keras==2.15.0`

If you use a different environment, ensure TF/Keras match the model training version, or use the weights fallback described below.

### Artifacts
Copy these files into `peakguard_api/artifacts/`:
- `series_minmax_scaler.pkl`
- Preferred: `gru_energy_forecaster.weights.h5`
- Optional: `gru_energy_forecaster.keras`

Tip (notebook): after training, save both
```python
model.save('artifacts/gru_energy_forecaster.keras')
model.save_weights('artifacts/gru_energy_forecaster.weights.h5')
```
The API tries to load the full model first; if deserialization fails (version mismatch), it reconstructs the architecture and loads `*.weights.h5`.

### Run
```
poetry run uvicorn main:app --reload --port 8000 --app-dir .
```
Open `http://127.0.0.1:8000`.

### Notebook training (Jupyter kernel)
- Install notebook tooling and register a kernel bound to the Poetry venv
  ```bash
  cd /Users/andressalguero/Documents/peakguard_api
  poetry install
  poetry run python -m ipykernel install --user --name peakguard-api --display-name "PeakGuard (poetry)"
  ```
- Set DB path (optional; defaults to `data/peakguard.db`)
  ```bash
  export DB_PATH=/Users/andressalguero/Documents/peakguard_api/data/peakguard.db
  ```
- Launch JupyterLab and open the training notebook
  ```bash
  poetry run jupyter lab
  ```
  Then open `experiments/TrainingPipeline.ipynb` and/or `experiments/H2OTrainingPipeline.ipynb` and select the kernel "PeakGuard (poetry)".

- From vscode/cursor use 
```bash
  poetry env info --path
```

Then Use <that_path>/bin/python as the kernel. 

### Endpoints
- `GET /api/devices`
  - Returns available devices: `[{ id, name, timezone }]`
- `GET /api/synthetic/latest?deviceId=<id>`
  - Returns the last 24 hours aligned to the device's timezone; data is always read from SQLite
- `POST /api/forecast`
  - Body: `{ "steps": 1, "deviceId": <id> }`
  - Returns only the next hour forecast using last 24h context for that device
  - Response payload includes:
    - `history` → last 24 hours
    - `forecast` → one timestamp (next hour) and one `y_pred`

### Frontend behavior
- On load, fetches `/api/devices` to populate the device selector
- Calls `/api/synthetic/latest?deviceId=...` and `POST /api/forecast` with `deviceId`
- Refreshes every hour (or on manual page refresh)
- Renders a 24h blue history line and a green segment connecting to the next-hour forecast point
- Live list shows 24 entries, scrollable, newest last

### Data storage (SQLite)
- Location: `DB_PATH` env var (default `/app/data/peakguard.db` in Docker)
- Tables:
  - `devices(id, name, timezone)`
  - `readings(id, device_id, ts_utc, consumption)` with `(device_id, ts_utc)` unique
  - `meta(key, value)` used to mark one-time seeding
- One-time seeding on first startup:
  - Inserts two devices: `Device A (America/New_York)` and `Device B (Europe/Berlin)`
  - Generates ~60 days of hourly synthetic data per device and persists to `readings`

### Background scheduler
- Runs continuously; wakes up each UTC hour and appends the next hour of synthetic readings for each device
- Endpoints never generate data on-demand; they only read from the DB
- For demo acceleration in local, you can add a configurable interval (open to implement `GENERATOR_INTERVAL_SECONDS`)

### Troubleshooting
- If you see Keras/TensorFlow deserialization errors when loading `.keras`, use the weights fallback:
  1) Ensure `gru_energy_forecaster.weights.h5` and `series_minmax_scaler.pkl` are present in `artifacts/`
  2) Restart the API; it will rebuild the model and load weights
- Make sure you run under Poetry’s environment (do not mix with other venvs)

### Docker & Railway
- The Docker image installs `tzdata` and sets `DB_PATH=/app/data/peakguard.db`
- Ensure your Railway deployment mounts a persistent volume at `/app/data` so the SQLite DB survives restarts
- Build & run locally:
  ```bash
  docker build -t peakguard-api .
  docker run --rm -p 8000:8000 -e PORT=8000 -e DB_PATH=/app/data/peakguard.db -v $(pwd)/data:/app/data peakguard-api
  ```

### H2O training (optional)
- Install H2O locally (requires JVM):
  ```bash
  poetry add h2o  # or: pip install h2o
  java -version   # verify Java is available
  ```
- Open and run `experiments/H2OTrainingPipeline.ipynb`:
  - Builds device-local cyclical features, lag features (1,2,3,6,12,24,48), 24h rolling mean
  - Trains H2O AutoML (regression) with an 80/20 time split
  - Saves MOJO to `artifacts/versions/<ts>/h2o/model.mojo.zip` and `artifacts/latest/h2o/model.mojo.zip`
  - Logs RMSE and a compact test-set plot in SQLite `models`/`model_results`
- Serving note: the API currently serves the Keras model. MOJO serving would require adding an H2O scoring runtime or Java-based scoring; we can add this later if desired.


