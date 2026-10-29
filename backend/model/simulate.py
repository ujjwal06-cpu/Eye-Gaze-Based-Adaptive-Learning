"""Simulate 60 days of gaze + request data for 3 patients.

    python -m model.simulate            # run from the backend/ directory

Outputs (default model/data/):
  requests.csv  patient_id, ts, request, source            (contract schema)
  gaze.csv      patient_id, ts, x, y, zone, dwell_ms, confidence   (contract schema)
  episodes.csv  patient_id, start, end, kind  (kind = focus|idle|confused)
                -- simulator ground truth, used only to tune/evaluate the
                   confusion threshold; not part of the contract.

Each gaze row is one fixation (ts = fixation onset). Gaze is generated in
windows rather than continuously:
  focus     90 s before every request, ~85% of dwell on the requested zone
  idle      4 calm 60 s windows per day wandering over 2-4 spots
  confused  3-8 min of scattered gaze (short dwells, many zone switches) on 5% of
            days, placed where a routine request would be expected; that
            request is NOT made.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .features import LABELS

DATA_DIR = Path(__file__).parent / "data"

# 3x2 grid of on-screen zones (x0, x1, y0, y1); gaps between them are "no zone".
ZONE_RECTS = {
    "water": (0.05, 0.31, 0.12, 0.45),
    "food": (0.37, 0.63, 0.12, 0.45),
    "help": (0.69, 0.95, 0.12, 0.45),
    "pain": (0.05, 0.31, 0.55, 0.88),
    "bathroom": (0.37, 0.63, 0.55, 0.88),
    "rest": (0.69, 0.95, 0.55, 0.88),
}

JITTER_MIN = 20          # +/- minutes of timing jitter
NOISE_P = 0.10           # fraction of requests replaced by a random label at a random time
CONFUSED_DAY_P = 0.05    # fraction of days with confused episodes
FOCUS_S = 90
IDLE_S = 60
IDLE_PER_DAY = 4
DAY_START_MIN, DAY_END_MIN = 7 * 60, 22 * 60


@dataclass
class Item:
    label: str
    minute: int                      # minute of day
    weekdays: tuple | None = None    # 0=Mon..6=Sun; None = every day


def hm(h: int, m: int = 0) -> int:
    return h * 60 + m


# Patient ids match backend/seed.py (1=Asha, 2=Ravi, 3=Meena).
ROUTINES = {
    1: [Item("water", hm(10)), Item("food", hm(13)), Item("rest", hm(15))],
    2: [
        Item("bathroom", hm(8, 30)), Item("water", hm(11, 30)), Item("food", hm(12, 45)),
        Item("help", hm(14), (0, 1, 2, 3, 4)),      # weekday physio
        Item("pain", hm(16, 30)), Item("rest", hm(20, 30)),
    ],
    3: [
        Item("food", hm(8)), Item("bathroom", hm(10, 30)),
        Item("pain", hm(11, 30), (5, 6)),           # weekends only
        Item("water", hm(14)), Item("help", hm(17, 30)),
        Item("food", hm(18, 30)), Item("rest", hm(21, 30)),
    ],
}


def zone_at(x: float, y: float):
    for label, (x0, x1, y0, y1) in ZONE_RECTS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return label
    return None


def centre(label: str):
    x0, x1, y0, y1 = ZONE_RECTS[label]
    return (x0 + x1) / 2, (y0 + y1) / 2


def clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


class Simulator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.gaze: list[tuple] = []
        self.requests: list[tuple] = []
        self.episodes: list[tuple] = []

    # ---- helpers -------------------------------------------------------
    def _emit(self, pid, t, x, y, dwell, conf):
        x, y = clip01(x), clip01(y)
        self.gaze.append((pid, t, x, y, zone_at(x, y), int(round(dwell)), conf))

    def _free(self, pid, start, end) -> bool:
        return all(
            end <= a or start >= b for p, a, b, _ in self.episodes if p == pid
        )

    # ---- gaze generators ----------------------------------------------
    def focus_window(self, pid, label, end):
        rng = self.rng
        start = end - timedelta(seconds=FOCUS_S)
        t = start
        while True:
            tgt = label if rng.random() < 0.85 else rng.choice([l for l in LABELS if l != label])
            cx, cy = centre(tgt)
            dwell = float(np.clip(rng.lognormal(math.log(900), 0.4), 300, 2500))
            if t + timedelta(milliseconds=dwell) > end:
                break
            self._emit(pid, t, rng.normal(cx, 0.04), rng.normal(cy, 0.04), dwell, rng.uniform(0.85, 0.99))
            t += timedelta(milliseconds=dwell + rng.uniform(40, 150))
        self.episodes.append((pid, start, end, "focus"))

    def idle_window(self, pid, start):
        rng = self.rng
        end = start + timedelta(seconds=IDLE_S)
        anchors = rng.uniform(0.05, 0.95, size=(int(rng.integers(2, 5)), 2))
        t = start
        while True:
            ax, ay = anchors[int(rng.integers(len(anchors)))]
            dwell = float(np.clip(rng.lognormal(math.log(1500), 0.4), 600, 4000))
            if t + timedelta(milliseconds=dwell) > end:
                break
            self._emit(pid, t, rng.normal(ax, 0.03), rng.normal(ay, 0.03), dwell, rng.uniform(0.8, 0.98))
            t += timedelta(milliseconds=dwell + rng.uniform(40, 150))
        self.episodes.append((pid, start, end, "idle"))

    def confused_episode(self, pid, start, dur_s):
        rng = self.rng
        end = start + timedelta(seconds=dur_s)
        t = start
        while t < end:
            if rng.random() < 0.7:  # jump between zones
                cx, cy = centre(LABELS[int(rng.integers(len(LABELS)))])
                x, y = rng.normal(cx, 0.09), rng.normal(cy, 0.09)
            else:                   # or anywhere on screen
                x, y = rng.uniform(0, 1), rng.uniform(0, 1)
            dwell = float(rng.uniform(80, 300))
            self._emit(pid, t, x, y, dwell, rng.uniform(0.6, 0.95))
            t += timedelta(milliseconds=dwell + rng.uniform(30, 100))
        self.episodes.append((pid, start, end, "confused"))

    # ---- one patient ---------------------------------------------------
    def run_patient(self, pid, items, start_day: datetime, n_days: int):
        rng = self.rng
        n_conf = round(CONFUSED_DAY_P * n_days)
        confused_days = set(rng.choice(n_days, size=n_conf, replace=False).tolist())

        for d in range(n_days):
            day = start_day + timedelta(days=d)

            # 1. routine occurrences with jitter; 10% become random noise
            occ = []
            for it in items:
                if it.weekdays is not None and day.weekday() not in it.weekdays:
                    continue
                t = day + timedelta(minutes=it.minute + rng.uniform(-JITTER_MIN, JITTER_MIN))
                occ.append({"t": t, "label": it.label, "noise": False})
            for o in occ:
                if rng.random() < NOISE_P:
                    o["noise"] = True
                    o["label"] = LABELS[int(rng.integers(len(LABELS)))]
                    o["t"] = day + timedelta(minutes=rng.uniform(DAY_START_MIN, DAY_END_MIN))

            # 2. confused day: scatter gaze where routine requests would be, make no request
            if d in confused_days:
                clean = [o for o in occ if not o["noise"]]
                picks = rng.choice(len(clean), size=min(2, len(clean)), replace=False)
                for i in picks:
                    o = clean[int(i)]
                    s = o["t"] - timedelta(seconds=60)
                    dur = float(rng.uniform(180, 480))
                    if self._free(pid, s, s + timedelta(seconds=dur)):
                        self.confused_episode(pid, s, dur)
                        o["dropped"] = True

            # 3. requests, each preceded by a focus window
            for o in occ:
                if o.get("dropped"):
                    continue
                t = o["t"]
                for _ in range(20):
                    if self._free(pid, t - timedelta(seconds=FOCUS_S), t):
                        break
                    if not o["noise"]:
                        t = None
                        break
                    t = day + timedelta(minutes=rng.uniform(DAY_START_MIN, DAY_END_MIN))
                else:
                    t = None
                if t is None:
                    continue
                self.focus_window(pid, o["label"], t)
                self.requests.append((pid, t, o["label"], "simulated"))

            # 4. calm idle windows
            for _ in range(IDLE_PER_DAY):
                for _try in range(10):
                    s = day + timedelta(minutes=rng.uniform(DAY_START_MIN, DAY_END_MIN))
                    if self._free(pid, s, s + timedelta(seconds=IDLE_S)):
                        self.idle_window(pid, s)
                        break


def fmt(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--start", default="2026-08-01", help="first day (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    sim = Simulator(np.random.default_rng(args.seed))
    start_day = datetime.fromisoformat(args.start)
    for pid, items in ROUTINES.items():
        sim.run_patient(pid, items, start_day, args.days)

    args.out.mkdir(parents=True, exist_ok=True)

    gaze = pd.DataFrame(sim.gaze, columns=["patient_id", "ts", "x", "y", "zone", "dwell_ms", "confidence"])
    gaze["ts"] = pd.to_datetime(gaze["ts"])
    gaze = gaze.sort_values(["patient_id", "ts"]).reset_index(drop=True)
    out = gaze.assign(ts=fmt(gaze["ts"]), x=gaze["x"].round(4), y=gaze["y"].round(4),
                      confidence=gaze["confidence"].round(3))
    out.to_csv(args.out / "gaze.csv", index=False)

    req = pd.DataFrame(sim.requests, columns=["patient_id", "ts", "request", "source"])
    req["ts"] = pd.to_datetime(req["ts"])
    req = req.sort_values(["patient_id", "ts"]).reset_index(drop=True)
    req.assign(ts=fmt(req["ts"])).to_csv(args.out / "requests.csv", index=False)

    ep = pd.DataFrame(sim.episodes, columns=["patient_id", "start", "end", "kind"])
    ep = ep.sort_values(["patient_id", "start"]).reset_index(drop=True)
    ep.assign(start=fmt(pd.to_datetime(ep["start"])), end=fmt(pd.to_datetime(ep["end"]))).to_csv(
        args.out / "episodes.csv", index=False
    )

    print(f"Wrote {args.out}/  requests={len(req)}  gaze={len(gaze)}  episodes={len(ep)}")
    for pid in ROUTINES:
        r = req[req.patient_id == pid]
        e = ep[(ep.patient_id == pid) & (ep.kind == "confused")]
        days = pd.to_datetime(e["start"]).dt.date.nunique()
        print(f"  patient {pid}: {len(r)} requests {r.request.value_counts().to_dict()}; "
              f"{len(e)} confused episodes on {days} days")


if __name__ == "__main__":
    main()
