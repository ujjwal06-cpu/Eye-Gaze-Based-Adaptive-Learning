"""PatientData: one patient's history as sorted numpy arrays, cheap to cut at any time t.

All times are epoch seconds (UTC). `view(t)` returns the history strictly before t, so the
same object serves live prediction and leak-free replay / evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import tzinfo

import numpy as np

from .config import LABEL_IDX, PATIENT_TZ, WIN_FIELDS, local_dt

FEEDBACK_ACTIONS = {"done": 0, "dismiss": 1}
FEEDBACK_NAMES = {v: k for k, v in FEEDBACK_ACTIONS.items()}
NONE_ZONE = -1


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _i(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def local_parts(ts: np.ndarray, tz: tzinfo = PATIENT_TZ) -> tuple[np.ndarray, np.ndarray]:
    """Minute of day and weekday (0 = Monday) in the patient's wall clock."""
    mod = np.empty(len(ts))
    wd = np.empty(len(ts), dtype=np.int64)
    for i, t in enumerate(ts):
        d = local_dt(float(t), tz)
        mod[i] = d.hour * 60 + d.minute + d.second / 60.0
        wd[i] = d.weekday()
    return mod, wd


@dataclass
class PatientData:
    req_t: np.ndarray = field(default_factory=lambda: _f([]))
    req_y: np.ndarray = field(default_factory=lambda: _i([]))
    req_mod: np.ndarray = field(default_factory=lambda: _f([]))
    req_wd: np.ndarray = field(default_factory=lambda: _i([]))
    win_t: np.ndarray = field(default_factory=lambda: _f([]))        # window START, epoch s
    win_X: np.ndarray = field(default_factory=lambda: np.zeros((0, len(WIN_FIELDS))))
    gaze_t: np.ndarray = field(default_factory=lambda: _f([]))
    gaze_z: np.ndarray = field(default_factory=lambda: _i([]))       # label index or -1
    gaze_d: np.ndarray = field(default_factory=lambda: _f([]))       # dwell_ms
    fb_t: np.ndarray = field(default_factory=lambda: _f([]))
    fb_y: np.ndarray = field(default_factory=lambda: _i([]))
    fb_a: np.ndarray = field(default_factory=lambda: _i([]))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_records(cls, requests=(), windows=(), gaze=(), feedback=(), tz: tzinfo = PATIENT_TZ):
        """requests [(t, label)], windows [(start_t, {field: value} | sequence of 8)],
        gaze [(t, zone|None, dwell_ms)], feedback [(t, label, action)]. Any order."""
        req = sorted((float(t), LABEL_IDX[l]) for t, l in requests if l in LABEL_IDX)
        r_t = _f([r[0] for r in req])
        mod, wd = local_parts(r_t, tz)

        wins = sorted(((float(t), _win_vec(x)) for t, x in windows), key=lambda w: w[0])
        g = sorted((float(t), LABEL_IDX.get(z, NONE_ZONE) if z else NONE_ZONE, float(d or 0)) for t, z, d in gaze)
        fb = sorted(
            (float(t), LABEL_IDX[l], FEEDBACK_ACTIONS[a])
            for t, l, a in feedback if l in LABEL_IDX and a in FEEDBACK_ACTIONS
        )
        return cls(
            req_t=r_t, req_y=_i([r[1] for r in req]), req_mod=mod, req_wd=wd,
            win_t=_f([w[0] for w in wins]),
            win_X=np.array([w[1] for w in wins]) if wins else np.zeros((0, len(WIN_FIELDS))),
            gaze_t=_f([x[0] for x in g]), gaze_z=_i([x[1] for x in g]), gaze_d=_f([x[2] for x in g]),
            fb_t=_f([x[0] for x in fb]), fb_y=_i([x[1] for x in fb]), fb_a=_i([x[2] for x in fb]),
        )

    def view(self, t: float) -> "PatientData":
        """Everything known strictly before t (a window counts once it has ended)."""
        r = int(np.searchsorted(self.req_t, t, side="left"))
        w = int(np.searchsorted(self.win_t, t - 10.0, side="right"))
        g = int(np.searchsorted(self.gaze_t, t, side="right"))
        f = int(np.searchsorted(self.fb_t, t, side="left"))
        return PatientData(
            self.req_t[:r], self.req_y[:r], self.req_mod[:r], self.req_wd[:r],
            self.win_t[:w], self.win_X[:w],
            self.gaze_t[:g], self.gaze_z[:g], self.gaze_d[:g],
            self.fb_t[:f], self.fb_y[:f], self.fb_a[:f],
        )

    # ---------------------------------------------------------------- append
    def add_request(self, t: float, label: str, tz: tzinfo = PATIENT_TZ) -> None:
        mod, wd = local_parts(_f([t]), tz)
        i = int(np.searchsorted(self.req_t, t, side="right"))
        self.req_t = np.insert(self.req_t, i, t)
        self.req_y = np.insert(self.req_y, i, LABEL_IDX[label])
        self.req_mod = np.insert(self.req_mod, i, mod[0])
        self.req_wd = np.insert(self.req_wd, i, wd[0])

    def add_window(self, t: float, values) -> None:
        i = int(np.searchsorted(self.win_t, t, side="right"))
        self.win_t = np.insert(self.win_t, i, t)
        self.win_X = np.insert(self.win_X, i, _win_vec(values), axis=0)

    def add_feedback(self, t: float, label: str, action: str) -> None:
        i = int(np.searchsorted(self.fb_t, t, side="right"))
        self.fb_t = np.insert(self.fb_t, i, t)
        self.fb_y = np.insert(self.fb_y, i, LABEL_IDX[label])
        self.fb_a = np.insert(self.fb_a, i, FEEDBACK_ACTIONS[action])

    @property
    def n_requests(self) -> int:
        return int(len(self.req_t))


def _win_vec(x) -> np.ndarray:
    if isinstance(x, dict):
        return np.array([float(x.get(f, 0.0) or 0.0) for f in WIN_FIELDS])
    v = np.asarray(x, dtype=float)
    if v.shape != (len(WIN_FIELDS),):
        raise ValueError(f"window needs {len(WIN_FIELDS)} values, got {v.shape}")
    return v
