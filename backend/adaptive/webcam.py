"""Patient state from the webcam (the 10 s B1 windows GazePad computes).

Each signal is compared with THIS patient's own baseline (median and MAD of their recent
windows), because "normal" head movement or blink rate differs a lot between patients.
Until 20 baseline windows exist, population defaults are used.

The three states are heuristics, not diagnoses:
  distress   more head movement, less stable gaze, more blinking, emergency presses
  fatigue    longer fixations, more blinking, less head movement
  confusion  unstable, wide-ranging gaze with short fixations
How much each state should move the need prediction is learned per patient by the ensemble
(webcam expert weight) and by the online model, so a wrong heuristic gets down-weighted.
"""
from __future__ import annotations

import math

import numpy as np

from .config import (
    BASELINE_LOOKBACK_S, BASELINE_MAX_WINDOWS, BASELINE_MIN_WINDOWS, POP_BASELINE,
    RECENT_WIN_S, W, WINDOW_S,
)
from .data import PatientData

# signal name -> (window field, transform). Heavy-tailed counts are log-transformed.
SIGNALS = {
    "head": ("Head_Movement", math.log1p),
    "stability": ("Gaze_Stability", None),
    "deviation": ("Mean_Gaze_Deviation", None),
    "blink": ("Blink_Rate", math.log1p),
    "fixation": ("Avg_Fixation_Duration", None),
}
SIGNAL_TEXT = {
    "head": "head movement", "stability": "gaze stability", "deviation": "gaze deviation",
    "blink": "blink rate", "fixation": "fixation length",
}
MIN_SCALE = {"head": 0.25, "stability": 0.02, "deviation": 0.5, "blink": 0.25, "fixation": 0.05}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _tf(fn, v):
    v = np.asarray(v, dtype=float)
    return np.log1p(np.maximum(v, 0)) if fn is math.log1p else v


def assess(h: PatientData, t: float) -> dict:
    """Webcam state at time t from windows that ended before t."""
    ends = h.win_t + WINDOW_S
    recent = (ends <= t) & (ends > t - RECENT_WIN_S)
    last_age = float(t - ends[-1]) if len(ends) else None
    if not recent.any():
        return {"available": False, "last_window_age_s": last_age, "n_recent": 0}

    R = h.win_X[recent]
    base_mask = (ends <= t - RECENT_WIN_S) & (ends > t - BASELINE_LOOKBACK_S)
    B = h.win_X[base_mask][-BASELINE_MAX_WINDOWS:]
    personal = len(B) >= BASELINE_MIN_WINDOWS

    z, now, base = {}, {}, {}
    for name, (fld, fn) in SIGNALS.items():
        r = _tf(fn, R[:, W[fld]])
        cur = float(np.median(r))
        if personal:
            b = _tf(fn, B[:, W[fld]])
            med = float(np.median(b))
            scale = float(1.4826 * np.median(np.abs(b - med)))
        else:
            m, s = POP_BASELINE[fld]
            med = float(_tf(fn, [m])[0])
            scale = float(_tf(fn, [m + s])[0] - med)
        scale = max(scale, MIN_SCALE[name])
        z[name] = float(np.clip((cur - med) / scale, -5, 5))
        now[name] = float(np.median(R[:, W[fld]]))
        base[name] = float(np.median(B[:, W[fld]])) if personal else POP_BASELINE[fld][0]

    emergency = int(R[:, W["Emergency_Button"]].sum())
    unstable = -z["stability"]  # positive = less stable than usual
    # distress = restless (head), confusion = wide-ranging gaze (deviation); both make gaze less stable
    distress = _sig(1.1 * z["head"] + 0.5 * unstable + 0.4 * z["blink"] + 2.5 * emergency - 1.8)
    fatigue = _sig(0.9 * z["fixation"] + 0.6 * z["blink"] - 0.5 * z["head"] - 1.8)
    confusion = _sig(1.0 * z["deviation"] + 0.5 * unstable - 0.5 * z["fixation"] - 0.4 * max(z["head"], 0) - 1.8)

    return {
        "available": True,
        "n_recent": int(recent.sum()),
        "last_window_age_s": last_age,
        "personal_baseline": personal,
        "baseline_windows": int(len(B)),
        "z": {k: round(v, 3) for k, v in z.items()},
        "now": {k: round(v, 3) for k, v in now.items()},
        "baseline": {k: round(v, 3) for k, v in base.items()},
        "emergency": emergency,
        "states": {"distress": round(distress, 3), "fatigue": round(fatigue, 3), "confusion": round(confusion, 3)},
    }


def describe(ws: dict, state: str) -> str:
    """Short human reason for a state, naming the signals that moved most."""
    z = ws.get("z", {})
    keys = {
        "distress": [("head", 1), ("stability", -1), ("blink", 1)],
        "fatigue": [("fixation", 1), ("blink", 1), ("head", -1)],
        "confusion": [("stability", -1), ("deviation", 1), ("fixation", -1)],
    }[state]
    parts = []
    for k, sign in keys:
        v = z.get(k, 0) * sign
        if v >= 1.0:
            word = "higher" if sign > 0 else "lower"
            if k == "fixation" and sign < 0:
                word = "shorter"
            elif k == "fixation":
                word = "longer"
            parts.append(f"{SIGNAL_TEXT[k]} {word} than usual")
    if state == "distress" and ws.get("emergency"):
        parts.insert(0, "Help tile selected")
    return ", ".join(parts[:3]) or "small changes in several signals"
