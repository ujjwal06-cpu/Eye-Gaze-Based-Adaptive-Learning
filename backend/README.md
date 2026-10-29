# Gaze Request API

FastAPI + SQLite (SQLAlchemy 2.0) + scikit-learn model. Implements the shared contract.

## Run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m model.simulate                # simulated data -> model/data/
python -m model.train                   # trains model -> model/artifacts/model.joblib
python seed.py --load-sim               # 3 patients (+ simulated history so /predict has context)
uvicorn app.main:app --reload --port 8000
```

Docs: http://localhost:8000/docs. Use `python seed.py` (no flag) for patients only.
Config env vars: `DATABASE_URL` (default `sqlite:///./app.db`), `CORS_ORIGINS` (default `*`).
Run everything from `backend/` so `model` is importable.

## Endpoints (contract)

| Method | Path | Notes |
|---|---|---|
| GET | `/patients` | |
| POST | `/gaze` | batch: JSON array of events, or `{"events": [...]}`; event = `{patient_id, ts, x, y, zone, dwell_ms, confidence}` |
| GET | `/gaze/recent?patient_id=&seconds=` | last N seconds up to server local time now; optional `until=` for a fixed end |
| POST | `/requests` | `{patient_id, ts, request, source}`; source defaults to `gaze` |
| GET | `/requests?patient_id=&from=&to=` | ascending by ts |
| POST | `/predict` | `{patient_id, ts}` -> `{suggestions:[{label, prob}] x3, confused, confusion_score}` |

Extras: `GET /health`, `GET /export/requests.csv?patient_id=` (columns `id,patient_id,ts,request,source`).

## Notes

- Labels: `water, food, help, pain, bathroom, rest`. Invalid labels return 422.
- `POST /predict` calls `predict()` from `model/predict.py`; history comes from this database via
  `app/db_source.py` (requests from the last 7 days, gaze from the last 30 s). It returns 503 if the model
  hasn't been trained.
- Timestamps are the patient's wall-clock time. Any timezone offset is dropped, not converted, matching the model.
- `gaze_events` and `requests` each have an index on `(patient_id, ts)`.
- Model details: `model/README.md`.

## Adaptive prediction and storage (new)

See `../README_ADAPTIVE.md` and `adaptive/README.md`. New endpoints (the ones above are unchanged):

| Method | Path | Notes |
|---|---|---|
| POST | `/windows` | B1 window envelope `{message, key, simulated, valid?, session_id?}` (or a list); duplicates ignored |
| GET | `/windows?patient_id=&minutes=` · `/windows/latest?patient_id=` | stored windows, exact B1 message |
| POST/PATCH | `/sessions`, `/sessions/{id}` | GazePad session with every selected parameter |
| POST | `/sessions/{id}/events` | calibration, setting changes, face lost/found, requests, end |
| GET | `/sessions?patient_id=` · `/sessions/{id}` | sessions, with event log |
| GET/POST | `/insights?patient_id=` | adaptive needs, reasons, webcam state, care actions, alert |
| POST/GET | `/feedback` | caregiver Done / Not needed |
| GET | `/model/stats?patient_id=` · `/model/evals` | out-of-sample accuracy over time, expert weights |
| POST | `/model/retrain?patient_id=` | replay history from scratch |
| POST | `/patients` | add a patient |
| GET | `/export/{windows,sessions,evals,feedback}.csv` | exports |

Patient ids `1`, `"1"`, `"p1"` and `"USER001"` are all accepted. Tests: `python -m pytest -q tests`.
Simulator: `python -m sim.ward evaluate | seed | live`.
