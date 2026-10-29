"""Generate requests and B1-shaped webcam windows for one virtual patient.

Windows are written every 2 min in the background and every 10 s (the real GazePad rate)
around anything interesting AND in random calm blocks, so window density is never a cue.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

import numpy as np

from adaptive.config import LABELS, PATIENT_TZ

from .personas import Persona

DEFAULT_BASE = {"deviation": 4.5, "head": 3.0, "stability": 0.33, "blink": 3.0, "fixation": 0.3}
DENSE_BEFORE_S = 5 * 60
BACKGROUND_EVERY_S = 120


@dataclass
class Signature:
    start: float
    end: float
    state: str
    peak: float
    visible: bool


def window_values(rng: np.random.Generator, base: dict, inten: dict) -> dict:
    """One 10 s window. inten: state -> 0..1 at this moment."""
    d, f, c = inten.get("distress", 0.0), inten.get("fatigue", 0.0), inten.get("confusion", 0.0)
    dev = base["deviation"] * (1 + 0.35 * d + 0.8 * c) + rng.normal(0, 0.7)
    dev = max(0.3, dev)
    maxdev = dev + abs(rng.normal(3.5 + 2 * c, 1.0))
    head = base["head"] * (1 + 3.0 * d + 0.6 * c - 0.5 * f) * rng.lognormal(0, 0.45)
    if rng.random() < 0.03:
        head *= rng.uniform(5, 25)          # the odd big head turn, as in real data
    stab = base["stability"] - 0.10 * d - 0.12 * c + 0.02 * f + rng.normal(0, 0.025)
    stab = float(np.clip(stab, 0.05, 0.95))
    blink = int(rng.poisson(max(0.2, base["blink"] * (1 + 0.6 * d + 1.0 * f + 0.2 * c))))
    fix = base["fixation"] * (1 - 0.25 * d + 1.3 * f - 0.45 * c) * rng.lognormal(0, 0.25)
    fix = float(np.clip(fix, 0.1, 6.0))
    maxfix = float(min(10.0, fix * rng.uniform(1.6, 3.5)))
    return {
        "Mean_Gaze_Deviation": round(dev, 3), "Max_Gaze_Deviation": round(maxdev, 3), "Blink_Rate": blink,
        "Avg_Fixation_Duration": round(fix, 3), "Max_Fixation_Duration": round(maxfix, 3),
        "Head_Movement": round(float(head), 3), "Gaze_Stability": round(stab, 3), "Emergency_Button": 0,
    }


def _pick(rng, dist: dict) -> str:
    labels, p = zip(*dist.items())
    return labels[int(rng.choice(len(labels), p=np.array(p) / sum(p)))]


def generate(persona: Persona, start: datetime, n_days: int, rng: np.random.Generator, tz: tzinfo = PATIENT_TZ,
             windows: bool = True, end_at: float | None = None) -> dict:
    """start: local midnight of the first day (aware or naive-local). Returns epoch-second events."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=tz)
    base = {**DEFAULT_BASE, **persona.baseline}
    requests, sigs = [], []
    aw0, aw1 = persona.awake

    for d in range(n_days):
        day = start + timedelta(days=d)
        t_day = day.timestamp()
        shift = persona.drift[1] if persona.drift and d >= int(persona.drift[0] * n_days) else 0

        for r in persona.routines:
            if r.weekdays is not None and day.weekday() not in r.weekdays:
                continue
            if rng.random() > r.prob:
                continue
            m = r.minute + shift + rng.uniform(-r.jitter, r.jitter)
            requests.append((t_day + m * 60, r.label, "routine"))

        for ep in persona.episodes:
            for _ in range(rng.poisson(ep.per_day)):
                m = rng.uniform(ep.hours[0] * 60, ep.hours[1] * 60)
                t = t_day + m * 60
                lead = rng.uniform(*ep.lead_min) * 60
                vis = rng.random() < ep.p_visible
                sigs.append(Signature(t - lead, t, ep.state, rng.uniform(0.6, 1.0), vis))
                if rng.random() < ep.p_request:
                    requests.append((t, _pick(rng, ep.labels), f"episode:{ep.state}"))

        for state, per_day in persona.spurious.items():
            for _ in range(rng.poisson(per_day)):
                t = t_day + rng.uniform(aw0, aw1) * 60
                sigs.append(Signature(t - rng.uniform(5, 15) * 60, t, state, rng.uniform(0.5, 0.9), True))

        for _ in range(rng.poisson(persona.noise_per_day)):
            requests.append((t_day + rng.uniform(aw0, aw1) * 60, LABELS[int(rng.integers(len(LABELS)))], "noise"))

    if end_at is not None:
        requests = [r for r in requests if r[0] <= end_at]
        sigs = [s for s in sigs if s.start <= end_at]
    requests.sort()
    # drop near-duplicates (two needs within 2 min): keep the first
    dedup = []
    for r in requests:
        if dedup and r[0] - dedup[-1][0] < 120:
            continue
        dedup.append(r)
    requests = dedup

    wins = _windows(persona, start, n_days, rng, base, requests, sigs, end_at) if windows else []
    return {"requests": requests, "windows": wins, "signatures": sigs, "baseline": base}


def intensity_at(sigs, t: float) -> dict:
    out = {}
    for s in sigs:
        if s.visible and s.start <= t <= s.end:
            v = s.peak * ((t - s.start) / max(1.0, s.end - s.start)) ** 1.5
            out[s.state] = max(out.get(s.state, 0.0), v)
    return out


def _windows(persona, start, n_days, rng, base, requests, sigs, end_at):
    aw0, aw1 = persona.awake
    times = set()
    for d in range(n_days):
        t0 = (start + timedelta(days=d)).timestamp()
        a, b = t0 + aw0 * 60, t0 + aw1 * 60
        times.update(np.arange(a, b, BACKGROUND_EVERY_S).round().tolist())
        for _ in range(4):  # calm dense blocks
            s = rng.uniform(a, b - DENSE_BEFORE_S)
            times.update(np.arange(s, s + DENSE_BEFORE_S, 10).round().tolist())
    for t, _, _ in requests:
        times.update(np.arange(t - DENSE_BEFORE_S, t - 10, 10).round().tolist())
    for s in sigs:
        times.update(np.arange(s.start - 60, s.end, 10).round().tolist())
    times = sorted(x for x in times if end_at is None or x + 10 <= end_at)

    req_t = np.array([r[0] for r in requests])
    req_help = np.array([r[1] == "help" for r in requests], dtype=bool)
    out = []
    sig_sorted = sorted(sigs, key=lambda s: s.start)
    starts = np.array([s.start for s in sig_sorted]) if sig_sorted else np.array([])
    for w in times:
        # only signatures that could be active (start <= w), and whose end is near
        k = int(np.searchsorted(starts, w, side="right"))
        active = [s for s in sig_sorted[max(0, k - 12):k] if s.end >= w]
        v = window_values(rng, base, intensity_at(active, w + 10))
        if len(req_t):  # a Help selection inside this window counts as Emergency_Button
            i = np.searchsorted(req_t, w)
            if i < len(req_t) and req_t[i] < w + 10 and req_help[i]:
                v["Emergency_Button"] = 1
        out.append((float(w), v))
    return out


def local_midnight(days_ago: int, tz: tzinfo = PATIENT_TZ) -> datetime:
    now = datetime.now(tz)
    return (now - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)


__all__ = ["generate", "window_values", "intensity_at", "local_midnight", "Signature"]
