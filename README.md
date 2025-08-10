## PeakGuard API

FastAPI service that simulates IoT energy consumption and serves GRU-based forecasts trained in `PeakGuard.ipynb`. The notebook is reference-only at runtime.

### Highlights
- Synthetic IoT stream (hourly) with browser-local time alignment
- Forecast endpoint returns only the next hour (using the last 24 hours as context)
- Responsive frontend (Plotly.js) showing 24h history and the next-hour forecast, rendered as a continuous segment
- Scrollable last-24-hours list (one entry per line)

### Project layout
```
peakguard_api/
  app/
    __init__.py
    main.py            # FastAPI app and routes
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
    PeakGuard.ipynb    # reference only
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

### Endpoints (browser-local time)
- `GET /api/synthetic/latest?nowMs=<epoch_ms>`
  - Returns the previous 24 hours ending at the hour specified by `nowMs` (milliseconds since epoch). If omitted, server local time is used.
- `POST /api/forecast`
  - Body: `{ "steps": 1, "nowMs": <epoch_ms> }`
  - Always returns only the next hour forecast (1 step), computed from the latest 24 hours ending at `nowMs`.
  - Response payload includes:
    - `history` → last 24 hours
    - `forecast` → one timestamp (next hour) and one `y_pred`

### Frontend behavior
- Calls `/api/synthetic/latest` and `/api/forecast` with `nowMs = Date.now()`
- Renders a 24h blue history line and a green segment connecting to the next-hour forecast point
- Live list shows 24 entries, scrollable, newest last

### Troubleshooting
- If you see Keras/TensorFlow deserialization errors when loading `.keras`, use the weights fallback:
  1) Ensure `gru_energy_forecaster.weights.h5` and `series_minmax_scaler.pkl` are present in `artifacts/`
  2) Restart the API; it will rebuild the model and load weights
- Make sure you run under Poetry’s environment (do not mix with other venvs)


