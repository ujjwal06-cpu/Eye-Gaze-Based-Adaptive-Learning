"""Constants shared by the adaptive engine. Labels and B1 field names are the existing contract."""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta, tzinfo

LABELS = ["water", "food", "help", "pain", "bathroom", "rest"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
N = len(LABELS)

LABEL_TEXT = {
    "water": "Water", "food": "Food", "help": "Help",
    "pain": "Pain", "bathroom": "Bathroom", "rest": "Rest",
}

# The 8 numeric fields of a B1 window (same names and order as b1-server/schema.js).
WIN_FIELDS = [
    "Mean_Gaze_Deviation", "Max_Gaze_Deviation", "Blink_Rate", "Avg_Fixation_Duration",
    "Max_Fixation_Duration", "Head_Movement", "Gaze_Stability", "Emergency_Button",
]
W = {f: i for i, f in enumerate(WIN_FIELDS)}

# Population defaults for a patient with too little webcam history (B1 spec "normal" values,
# nudged by what real GazePad windows looked like). Used only until a personal baseline exists.
POP_BASELINE = {
    "Mean_Gaze_Deviation": (4.5, 1.2),
    "Blink_Rate": (3.0, 1.5),
    "Avg_Fixation_Duration": (0.30, 0.12),
    "Head_Movement": (6.0, 5.0),
    "Gaze_Stability": (0.33, 0.05),
}

# --- timing -------------------------------------------------------------------------
LEAD_S = 15.0                 # prequential evaluation: predict this long before each request
WINDOW_S = 10.0               # a B1 window covers 10 s
RECENT_WIN_S = 90.0           # windows that ended in the last 90 s describe "now"
BASELINE_MAX_WINDOWS = 720    # personal baseline: up to the last 720 windows ...
BASELINE_LOOKBACK_S = 24 * 3600   # ... from the last 24 h
BASELINE_MIN_WINDOWS = 20
GAZE_FOCUS_S = 30.0
HISTORY_DAYS = 60

# --- experts ------------------------------------------------------------------------
ROUTINE_SIGMA_MIN = 35.0      # time-of-day kernel width
ROUTINE_DECAY_DAYS = 10.0     # older requests count less, so changed routines are re-learned
TRANSITION_GAP_S = 4 * 3600   # A -> B counts as a transition if B follows within 4 h
DUE_MIN_GAP_S = 10 * 60
DUE_MAX_GAP_S = 36 * 3600

# --- ensemble -----------------------------------------------------------------------
ETA = 0.7                     # Hedge learning rate (log loss)
FIXED_SHARE = 0.02            # keeps every expert re-learnable after a routine change
EXPERT_SMOOTH = 0.08          # mix each expert with uniform so no label gets probability 0
PRIOR_WEIGHTS = {
    "routine": 1.0, "transition": 0.6, "due": 0.8, "webcam": 0.8,
    "gaze_focus": 1.0, "learned": 0.5, "learned_global": 0.5, "static_rf": 0.6,
}

# --- caregiver feedback -------------------------------------------------------------
FEEDBACK_DONE_S = 30 * 60     # "done" lowers that need for 30 min
FEEDBACK_DISMISS_S = 45 * 60  # "not needed" lowers it for 45 min
FEEDBACK_FACTOR = {"done": 0.4, "dismiss": 0.35}


def _resolve_tz() -> tzinfo:
    """Patient wall-clock zone for time-of-day features. PATIENT_TZ (IANA name) or the server's zone."""
    name = os.getenv("PATIENT_TZ")
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:  # unknown name: fall back to the server zone
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


PATIENT_TZ = _resolve_tz()


def local_dt(t: float, tz: tzinfo | None = None) -> datetime:
    return datetime.fromtimestamp(t, tz or PATIENT_TZ)


def to_epoch(dt) -> float:
    """Naive datetimes in this project are UTC (GazePad sends toISOString())."""
    if isinstance(dt, (int, float)):
        return float(dt)
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def from_epoch(t: float) -> datetime:
    """Epoch seconds -> naive UTC datetime (the database convention)."""
    return datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None)


def fmt_clock(t: float) -> str:
    return local_dt(t).strftime("%H:%M")


def fmt_minute_of_day(m: float) -> str:
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def fmt_duration(s: float) -> str:
    s = max(0, int(round(s)))
    if s < 90:
        return f"{s} s"
    m = s // 60
    if m < 90:
        return f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min" if m else f"{h} h"


ONE_DAY = timedelta(days=1)
