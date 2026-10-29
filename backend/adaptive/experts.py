"""Experts: each looks at one kind of evidence and returns P(label | a request now), or None
when it has nothing to say (a "sleeping" expert is ignored by the ensemble and not penalised).

    routine      what this patient usually asks for at this time of day (recency weighted)
    transition   what usually follows their last request
    due          needs that are overdue compared with their usual interval
    webcam       distress / fatigue / confusion from the B1 webcam windows
    gaze_focus   tiles they have been looking at in the last 30 s without selecting
    learned      online log-linear pool per patient: learns how far to trust each expert and
                 what this patient's webcam states mean (adaptive/online.py)
    learned_global   the same pool shared by all patients (cold start for new patients)
    static_rf    the existing RandomForest from model/ (if trained), as one more opinion
"""
from __future__ import annotations

import math

import numpy as np

from .config import (
    DUE_MAX_GAP_S, DUE_MIN_GAP_S, GAZE_FOCUS_S, LABEL_TEXT, LABELS, N, ROUTINE_DECAY_DAYS,
    ROUTINE_SIGMA_MIN, TRANSITION_GAP_S, fmt_duration, fmt_minute_of_day,
)
from .data import NONE_ZONE, PatientData
from . import webcam as wc

UNIFORM = np.full(N, 1.0 / N)


def _norm(v) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 0)
    s = v.sum()
    return v / s if s > 0 else UNIFORM.copy()


def _circ(a, b):
    d = np.abs(np.asarray(a) - b) % 1440.0
    return np.minimum(d, 1440.0 - d)


def _reason(expert, label, text, strength):
    return {"expert": expert, "label": label, "text": text, "strength": round(float(strength), 3)}


# ------------------------------------------------------------------------- routine
def routine(h: PatientData, t: float, ctx: dict):
    n = h.n_requests
    if n < 3:
        return None, []
    age_d = (t - h.req_t) / 86400.0
    decay = np.exp(-age_d / ROUTINE_DECAY_DAYS)
    d = _circ(h.req_mod, ctx["mod"])
    kern = np.exp(-0.5 * (d / ROUTINE_SIGMA_MIN) ** 2)
    weekend = ctx["wd"] >= 5
    same_type = np.where((h.req_wd >= 5) == weekend, 1.0, 0.5)
    w = kern * decay * same_type
    score = np.bincount(h.req_y, weights=w, minlength=N)
    marg = np.bincount(h.req_y, weights=decay, minlength=N)
    p = _norm(score + 0.3 * _norm(marg) + 1e-3)

    reasons = []
    top = int(np.argmax(p))
    if score.sum() > 0.5 and p[top] > 0.3:
        m = (h.req_y == top) & (d <= 75) & (age_d <= 14)
        if m.any():
            days = len(set(np.floor(age_d[m]).astype(int).tolist()))
            seen_days = max(days, len(set(np.floor(age_d[age_d <= 14]).astype(int).tolist())))
            ang = h.req_mod[m] / 1440 * 2 * math.pi
            usual = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * math.pi) * 1440) % 1440
            reasons.append(_reason(
                "routine", LABELS[top],
                f"Usually asks for {LABEL_TEXT[LABELS[top]]} around {fmt_minute_of_day(usual)} "
                f"({days} of the last {seen_days} days)", p[top]))
    return p, reasons


def upcoming(h: PatientData, t: float, ctx: dict, horizon_min: int = 120, min_frac: float = 0.4):
    """Needs that usually come up in the next `horizon_min` minutes (for proactive care)."""
    if h.n_requests < 5:
        return []
    age_d = (t - h.req_t) / 86400.0
    recent = age_d <= 14
    if not recent.any():
        return []
    day_idx = np.floor(age_d[recent]).astype(int)
    n_days = max(1, len(set(day_idx.tolist())))
    out = []
    for c in range(N):
        mask = recent & (h.req_y == c)
        if mask.sum() < 2:
            continue
        if ((h.req_y == c) & (t - h.req_t < 45 * 60)).any():
            continue  # asked for it in the last 45 min already
        mods, days = h.req_mod[mask], np.floor(age_d[mask]).astype(int)
        best = None
        for k in range(5, horizon_min + 1, 5):
            m = (ctx["mod"] + k) % 1440
            frac = len(set(days[_circ(mods, m) <= 30].tolist())) / n_days
            if frac >= min_frac and (best is None or frac > best[1] + 0.05):
                best = (k, frac)
        if best:
            out.append({
                "label": LABELS[c], "in_min": best[0], "at": t + best[0] * 60,
                "usual_fraction": round(best[1], 2),
                "text": f"{LABEL_TEXT[LABELS[c]]} usually requested in about {best[0]} min "
                        f"({round(best[1] * 100)}% of recent days)",
            })
    return sorted(out, key=lambda u: u["in_min"])


# ---------------------------------------------------------------------- transition
def transition(h: PatientData, t: float, ctx: dict):
    n = h.n_requests
    if n < 5:
        return None, []
    gap = t - h.req_t[-1]
    if gap > 2 * TRANSITION_GAP_S:
        return None, []
    a, b = h.req_y[:-1], h.req_y[1:]
    ok = (h.req_t[1:] - h.req_t[:-1]) <= TRANSITION_GAP_S
    decay = np.exp(-((t - h.req_t[1:]) / 86400.0) / (2 * ROUTINE_DECAY_DAYS))
    last = int(h.req_y[-1])
    sel = ok & (a == last)
    counts = np.bincount(b[sel], weights=decay[sel], minlength=N)
    marg = _norm(np.bincount(h.req_y, minlength=N) + 0.5)
    p_trans = _norm(counts + 1.0 * marg)
    lam = math.exp(-gap / (3 * 3600))
    p = lam * p_trans + (1 - lam) * marg
    reasons = []
    top = int(np.argmax(p_trans))
    k = int(sel.sum())
    if lam > 0.3 and k >= 3 and p_trans[top] > 0.35:
        hits = int((b[sel] == top).sum())
        reasons.append(_reason(
            "transition", LABELS[top],
            f"After {LABEL_TEXT[LABELS[last]]}, usually asks for {LABEL_TEXT[LABELS[top]]} next ({hits} of {k} times)",
            p[top]))
    return p, reasons


# ----------------------------------------------------------------------------- due
def due(h: PatientData, t: float, ctx: dict):
    if h.n_requests < 6:
        return None, []
    recent = (t - h.req_t) <= 30 * 86400
    score = np.full(N, 0.15)
    info, any_label = {}, False
    for c in range(N):
        ts = h.req_t[recent & (h.req_y == c)]
        if len(ts) < 3:
            continue
        g = np.diff(ts)
        g = g[(g >= DUE_MIN_GAP_S) & (g <= DUE_MAX_GAP_S)]
        if len(g) < 2:
            continue
        any_label = True
        med = float(np.median(g))
        spread = max(0.25 * med, 15 * 60)
        since = t - ts[-1]
        score[c] = 0.05 + 1.0 / (1.0 + math.exp(-(since - med) / spread))
        if since < 10 * 60:
            score[c] *= 0.3  # just asked for it
        info[c] = (since, med)
    if not any_label:
        return None, []
    p = _norm(score)
    reasons = []
    for c, (since, med) in sorted(info.items(), key=lambda kv: -p[kv[0]]):
        if since >= 0.9 * med and p[c] > 0.2:
            reasons.append(_reason(
                "due", LABELS[c],
                f"Last {LABEL_TEXT[LABELS[c]]} {fmt_duration(since)} ago; usually every ~{fmt_duration(med)}",
                p[c]))
            break
    return p, reasons


# -------------------------------------------------------------------------- webcam
# need profile of each state, over LABELS = water, food, help, pain, bathroom, rest
STATE_NEEDS = {
    "distress": np.array([0.05, 0.03, 0.35, 0.45, 0.12, 0.00]),
    "fatigue": np.array([0.12, 0.05, 0.02, 0.04, 0.05, 0.72]),
    "confusion": np.array([0.10, 0.05, 0.55, 0.10, 0.10, 0.10]),
}
STATE_TEXT = {"distress": "signs of discomfort", "fatigue": "signs of tiredness", "confusion": "scattered, unsettled gaze"}


def webcam(h: PatientData, t: float, ctx: dict):
    ws = ctx["webcam"]
    if not ws.get("available"):
        return None, []
    st = ws["states"]
    if ws.get("emergency"):
        st = {**st, "distress": max(st["distress"], 0.9)}
    strongest = max(st, key=st.get)
    if st[strongest] < 0.4:
        return None, []  # calm: no opinion (specialist expert)
    v = 0.25 * UNIFORM
    for k, s in st.items():
        v = v + s * STATE_NEEDS[k]
    p = _norm(v)
    reasons = [
        _reason("webcam", None, f"Webcam: {STATE_TEXT[k]} ({wc.describe(ws, k)})", s)
        for k, s in sorted(st.items(), key=lambda kv: -kv[1]) if s >= 0.5
    ]
    return p, reasons


# ---------------------------------------------------------------------- gaze focus
def gaze_focus(h: PatientData, t: float, ctx: dict):
    m = (h.gaze_t > t - GAZE_FOCUS_S) & (h.gaze_t <= t) & (h.gaze_z != NONE_ZONE)
    if h.n_requests:  # ignore the dwell that produced the last request and the moment after it
        lt, ly = h.req_t[-1], h.req_y[-1]
        m &= ~((h.gaze_z == ly) & (h.gaze_t > lt - 3) & (h.gaze_t < lt + 5))
    if m.sum() < 15:
        return None, []
    z = h.gaze_z[m]
    share = np.bincount(z, minlength=N).astype(float)
    share /= share.sum()
    top = int(np.argmax(share))
    if share[top] < 0.35:
        return None, []
    p = _norm(0.15 * UNIFORM + 0.85 * share)
    return p, [_reason(
        "gaze_focus", LABELS[top],
        f"Looking at the {LABEL_TEXT[LABELS[top]]} tile ({round(share[top] * 100)}% of the last 30 s) without selecting it",
        p[top])]
