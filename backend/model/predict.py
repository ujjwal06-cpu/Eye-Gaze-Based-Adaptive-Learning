"""Model entry point used by POST /predict.

    predict(patient_id, ts) -> {
        "suggestions": [{"label": str, "prob": float}, ...],   # top 3
        "confused": bool,
        "confusion_score": float,                              # normalised gaze entropy, 0-1
    }

Behaviour
- confusion_score = dwell-weighted gaze entropy over the last 30 s (log base 7, so 0-1).
- confused = score > tuned threshold AND at least `min_events` gaze events in the window.
- Not confused -> top 3 from the patient's RandomForest (global model for unknown patients).
- Confused    -> top 3 from the time-of-day prior (what this patient usually asks for
                 around this time, e.g. water at 10am), ignoring the model.

`ts` is treated as the patient's wall-clock time: a timezone, if present, is dropped
without conversion (the same convention as training).

Where history comes from
- The model needs a patient's past requests and last 30 s of gaze. By default these are
  read from model/data/*.csv (the simulated data). The backend should call
  `set_data_source(...)` with an object that reads its database instead:

      class MySource:
          def requests(self, patient_id, until):
              # -> (ts: datetime64[ns] ascending, codes: int array of label indices),
              #    requests before `until`; more rows are harmless
          def gaze(self, patient_id, since, until):
              # -> (ts: datetime64[ns] ascending, zone codes, dwell_ms)
              #    for since < ts <= until; use features.zone_codes() to encode zones

Train first:  python -m model.train
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import (
    ENTROPY_IDX, LABEL_IDX, LABELS, WINDOW_S, build_features, minute_of_day, time_prior, zone_codes,
)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "artifacts" / "model.joblib"
DATA_DIR = HERE / "data"

_bundle = None
_source = None


class CSVSource:
    """Default history source: model/data/requests.csv and gaze.csv (loaded once)."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._req = self._gaze = None

    def _load(self):
        if self._req is None:
            req = pd.read_csv(self.data_dir / "requests.csv")
            gaze = pd.read_csv(self.data_dir / "gaze.csv")
            req["ts"] = pd.to_datetime(req["ts"]).astype("datetime64[ns]")
            gaze["ts"] = pd.to_datetime(gaze["ts"]).astype("datetime64[ns]")
            self._req = {p: d.sort_values("ts") for p, d in req.groupby("patient_id")}
            self._gaze = {p: d.sort_values("ts") for p, d in gaze.groupby("patient_id")}

    def requests(self, patient_id, until):
        self._load()
        d = self._req.get(patient_id)
        if d is None:
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.int64)
        return d.ts.to_numpy("datetime64[ns]"), np.array([LABEL_IDX[l] for l in d.request], dtype=np.int64)

    def gaze(self, patient_id, since, until):
        self._load()
        d = self._gaze.get(patient_id)
        if d is None:
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.int64), np.array([])
        d = d[(d.ts > since) & (d.ts <= until)]
        return d.ts.to_numpy("datetime64[ns]"), zone_codes(d.zone), d.dwell_ms.to_numpy(float)


def set_data_source(source) -> None:
    """Plug in a history source (e.g. one that reads the backend database)."""
    global _source
    _source = source


def reload() -> None:
    """Drop cached model/data so the next predict() reloads from disk."""
    global _bundle, _source
    _bundle = None
    if isinstance(_source, CSVSource):
        _source = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m model.simulate && python -m model.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _get_source():
    global _source
    if _source is None:
        _source = CSVSource()
    return _source


def _wall_clock(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _full_proba(model, x) -> np.ndarray:
    out = np.zeros(len(LABELS))
    out[model.classes_.astype(int)] = model.predict_proba(x.reshape(1, -1))[0]
    return out


def predict(patient_id: int, ts) -> dict:
    b = _get_bundle()
    src = _get_source()
    ts = _wall_clock(ts)
    patient_id = int(patient_id)

    req_ts, req_codes = src.requests(patient_id, ts)
    g_ts, g_zone, g_dwell = src.gaze(patient_id, ts - timedelta(seconds=WINDOW_S), ts)
    x, n_events = build_features(ts, req_ts, req_codes, g_ts, g_zone, g_dwell)

    score = float(x[ENTROPY_IDX])
    confused = bool(n_events >= b["min_events"] and score > b["threshold"])

    if confused:
        prior = b["prior"].get(patient_id, b["prior_global"])
        probs = time_prior(prior, minute_of_day(ts), b["prior_sigma_min"])
    else:
        model = b["models"].get(patient_id, b["global_model"])
        probs = _full_proba(model, x)

    top = np.argsort(-probs)[:3]
    return {
        "suggestions": [{"label": LABELS[i], "prob": round(float(probs[i]), 4)} for i in top],
        "confused": confused,
        "confusion_score": round(score, 4),
    }
