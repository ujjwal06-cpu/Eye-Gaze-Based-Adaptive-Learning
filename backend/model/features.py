"""Feature extraction shared by train.py and predict.py (single source of truth).

Conventions
- Timestamps are treated as the patient's wall-clock time; hour/minute/day_of_week
  are read from `ts` exactly as given (no timezone conversion).
- Gaze `zone` is one of LABELS or null. Null is its own category ("none") so that
  looking at empty screen space still counts toward entropy / zone switches.
- gaze_entropy_30s is dwell-weighted Shannon entropy over the 7 zone categories,
  using log base 7, so it lies in [0, 1]. It is also the `confusion_score`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LABELS = ["water", "food", "help", "pain", "bathroom", "rest"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
NONE_ZONE = len(LABELS)          # zone code for zone == null
N_ZONE_CODES = len(LABELS) + 1

WINDOW_S = 30                    # gaze lookback window
MAX_MINS = 1440.0                # cap for "minutes since last <label>" (24h)
MIN_EVENTS = 5                   # fewer gaze events than this -> never "confused"
PRIOR_SIGMA_MIN = 30.0           # time-of-day prior kernel width (minutes)

FEATURE_NAMES = (
    ["hour", "minute", "day_of_week"]
    + [f"mins_since_{l}" for l in LABELS]
    + [f"last_is_{l}" for l in LABELS]
    + ["zone_switches_30s", "gaze_entropy_30s", "mean_dwell_30s"]
)
ENTROPY_IDX = FEATURE_NAMES.index("gaze_entropy_30s")


def zone_codes(zones) -> np.ndarray:
    """Map zone strings (or NaN/None) to integer codes."""
    return np.array(
        [LABEL_IDX.get(z, NONE_ZONE) if isinstance(z, str) else NONE_ZONE for z in zones],
        dtype=np.int64,
    )


def to_ns(ts) -> np.datetime64:
    return pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]")


def gaze_window_stats(g_ts, g_zone, g_dwell, ts):
    """Stats over gaze events with ts-30s < event_ts <= ts.

    g_ts must be ascending datetime64[ns]. Returns
    (zone_switches, entropy[0..1], mean_dwell_ms, n_events).
    """
    t = to_ns(ts)
    lo = np.searchsorted(g_ts, t - np.timedelta64(WINDOW_S, "s"), side="right")
    hi = np.searchsorted(g_ts, t, side="right")
    n = int(hi - lo)
    if n == 0:
        return 0, 0.0, 0.0, 0
    z = g_zone[lo:hi]
    d = np.asarray(g_dwell[lo:hi], dtype=float)
    switches = int(np.count_nonzero(z[1:] != z[:-1]))
    w = np.bincount(z, weights=d, minlength=N_ZONE_CODES)
    if w.sum() <= 0:
        w = np.bincount(z, minlength=N_ZONE_CODES).astype(float)
    p = w[w > 0] / w.sum()
    entropy = float(-(p * np.log(p)).sum() / math.log(N_ZONE_CODES))
    return switches, entropy, float(d.mean()), n


def build_features(ts, req_ts, req_codes, g_ts, g_zone, g_dwell):
    """Feature vector (ordered as FEATURE_NAMES) at time `ts`, plus gaze event count.

    Only requests strictly before `ts` are used. req_ts must be ascending.
    """
    ts = pd.Timestamp(ts)
    t = to_ns(ts)
    mins = np.full(len(LABELS), MAX_MINS)
    last = np.zeros(len(LABELS))
    idx = int(np.searchsorted(req_ts, t, side="left"))
    if idx > 0:
        pts, pc = req_ts[:idx], req_codes[:idx]
        for c in range(len(LABELS)):
            hits = np.flatnonzero(pc == c)
            if hits.size:
                mins[c] = min(MAX_MINS, float((t - pts[hits[-1]]) / np.timedelta64(1, "m")))
        last[pc[-1]] = 1.0
    switches, entropy, mean_dwell, n = gaze_window_stats(g_ts, g_zone, g_dwell, ts)
    vec = np.concatenate(
        [[ts.hour, ts.minute, ts.dayofweek], mins, last, [switches, entropy, mean_dwell]]
    ).astype(float)
    return vec, n


def minute_of_day(ts) -> float:
    ts = pd.Timestamp(ts)
    return ts.hour * 60 + ts.minute + ts.second / 60.0


def time_prior(minutes_by_label: dict, minute: float, sigma: float = PRIOR_SIGMA_MIN) -> np.ndarray:
    """P(label | time of day) from a circular Gaussian kernel over past request times.

    minutes_by_label maps label code -> array of minute-of-day values of past requests.
    """
    w = np.full(len(LABELS), 1e-3)
    for c, mins in minutes_by_label.items():
        mins = np.asarray(mins, dtype=float)
        if mins.size:
            d = np.abs(mins - minute)
            d = np.minimum(d, 1440.0 - d)
            w[c] += np.exp(-0.5 * (d / sigma) ** 2).sum()
    return w / w.sum()
