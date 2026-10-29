"""Ward simulation for gaze-based adaptive need prediction.

Run from backend/:

    python -m sim.ward evaluate [--days 21] [--seed 1]
        Offline, no database. Replays virtual patients day by day through the adaptive
        engine (prequential: every prediction is made before the request is seen) and prints
        the learning curve against baselines. Report -> sim/reports/.

    python -m sim.ward seed [--days 14] [--reset]
        Writes virtual patients, their requests and webcam windows (ending now) into the
        backend database and trains their adaptive state, so the dashboard has history.

    python -m sim.ward live [--api http://localhost:8000] [--speed 1]
        Streams live webcam windows every 10 s and requests through the same HTTP endpoints
        GazePad uses, so the dashboard updates in real time. Needs `seed` first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adaptive import Engine, PatientData
from adaptive.config import LABELS, PATIENT_TZ, local_dt

from .generator import generate, intensity_at, local_midnight, window_values, Signature
from .personas import BY_KEY, PERSONAS

REPORT_DIR = Path(__file__).parent / "reports"
SIM_TAG = "[sim:"


# ======================================================================= evaluate
def _frequency_baseline():
    counts = defaultdict(lambda: np.ones(len(LABELS)))

    def predict_and_learn(pid, label):
        c = counts[pid]
        p = c / c.sum()
        y = LABELS.index(label)
        rank = int((p >= p[y] - 1e-12).sum())
        c *= 0.98
        c[y] += 1
        return rank

    return predict_and_learn


def evaluate(days: int, seed: int, quiet: bool = False, write: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    start = local_midnight(days)
    patients = {}
    for i, p in enumerate(PERSONAS):
        g = generate(p, start, days, rng)
        data = PatientData.from_records(requests=[(t, l) for t, l, _ in g["requests"]], windows=g["windows"])
        patients[i] = {"persona": p, "gen": g, "data": data}

    configs = {
        "adaptive": Engine(),
        "no_webcam": Engine(use_webcam=False),
        "routine_only": Engine(use_webcam=False, disabled={"transition", "due", "gaze_focus", "learned", "learned_global"}),
    }
    states = {k: {pid: e.new_state() for pid in patients} for k, e in configs.items()}
    gstates = {k: Engine.new_global_state() for k in configs}
    freq = _frequency_baseline()

    stream = sorted((t, pid, l, cause) for pid, P in patients.items() for t, l, cause in P["gen"]["requests"])
    t_first = stream[0][0]
    rows = []
    t0 = time.time()
    for t, pid, label, cause in stream:
        row = {"day": int((t - t_first) // 86400) + 1, "pid": pid, "persona": patients[pid]["persona"].key,
               "label": label, "cause": cause.split(":")[0], "cause_detail": cause, "t": t}
        for k, eng in configs.items():
            rec = eng.learn(patients[pid]["data"], states[k][pid], t, label,
                            gstates[k] if k != "routine_only" else None, patient_id=None)
            row[f"{k}_rank"] = rec["rank"]
            row[f"{k}_p"] = rec["p_actual"]
        row["frequency_rank"] = freq(pid, label)
        rows.append(row)
    elapsed = time.time() - t0

    names = list(configs) + ["frequency"]

    def summ(rs):
        n = len(rs)
        out = {"n": n}
        for k in names:
            out[f"{k}_top1"] = round(sum(r[f"{k}_rank"] == 1 for r in rs) / n, 3) if n else None
            out[f"{k}_top3"] = round(sum(r[f"{k}_rank"] <= 3 for r in rs) / n, 3) if n else None
        for k in configs:
            out[f"{k}_logloss"] = round(-sum(math.log(max(r[f"{k}_p"], 1e-6)) for r in rs) / n, 3) if n else None
        return out

    by_day = [{"day": d, **summ([r for r in rows if r["day"] == d])} for d in range(1, days + 1)]
    by_week = [{"days": f"{w * 7 + 1}-{min(days, w * 7 + 7)}", **summ([r for r in rows if w * 7 < r["day"] <= w * 7 + 7])}
               for w in range(math.ceil(days / 7))]
    by_persona = {p.key: summ([r for r in rows if r["persona"] == p.key]) for p in PERSONAS}
    by_cause = {c: summ([r for r in rows if r["cause"] == c]) for c in ("routine", "episode", "noise")}
    by_episode = {c: summ([r for r in rows if r["cause_detail"] == c])
                  for c in sorted({r["cause_detail"] for r in rows if r["cause"] == "episode"})}
    weights = {patients[pid]["persona"].key: _norm_weights(s["weights"]) for pid, s in states["adaptive"].items()}

    # adaptation after a routine change (stroke persona): accuracy in the 3 days after the drift
    drift = {}
    for pid, P in patients.items():
        p = P["persona"]
        if p.drift:
            dday = int(p.drift[0] * days) + 1
            rs = [r for r in rows if r["pid"] == pid]
            drift[p.key] = {
                "change_on_day": dday,
                "before (3 days)": summ([r for r in rs if dday - 3 <= r["day"] < dday]),
                "days 1-3 after": summ([r for r in rs if dday <= r["day"] < dday + 3]),
                "days 4-7 after": summ([r for r in rs if dday + 3 <= r["day"] < dday + 7]),
            }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": days, "seed": seed, "patients": [p.key for p in PERSONAS], "requests": len(rows),
        "replay_seconds": round(elapsed, 1), "overall": summ(rows), "by_week": by_week, "by_day": by_day,
        "by_persona": by_persona, "by_cause": by_cause, "by_episode": by_episode,
        "routine_change": drift, "final_expert_weights": weights,
        "note": "Prequential: each prediction is made 15 s before the request using only earlier data.",
    }
    if not write:
        return report
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    with open(REPORT_DIR / "evaluation_requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if not quiet:
        _print_report(report, names)
    return report


def _norm_weights(w):
    s = sum(w.values()) or 1
    return {k: round(v / s, 3) for k, v in sorted(w.items(), key=lambda kv: -kv[1])}


def _print_report(rep, names):
    print(f"\n{rep['requests']} requests from {len(rep['patients'])} virtual patients over {rep['days']} days "
          f"(replayed in {rep['replay_seconds']} s)\n")
    hdr = "".join(f"{k:>16}" for k in names)
    print(f"{'top-1 / top-3':<14}{hdr}")
    for wk in rep["by_week"]:
        print(f"days {wk['days']:<9}" + "".join(f"{wk[k + '_top1']:>9.2f} /{wk[k + '_top3']:.2f}" for k in names))
    o = rep["overall"]
    print(f"{'overall':<14}" + "".join(f"{o[k + '_top1']:>9.2f} /{o[k + '_top3']:.2f}" for k in names))
    print("\nby cause (adaptive vs no_webcam, top-1):")
    for c, s in rep["by_cause"].items():
        if s["n"]:
            print(f"  {c:<10} n={s['n']:<4} adaptive {s['adaptive_top1']:.2f}   no_webcam {s['no_webcam_top1']:.2f}   "
                  f"routine_only {s['routine_only_top1']:.2f}")
    for c, s in rep["by_episode"].items():
        print(f"  {c:<18} n={s['n']:<4} adaptive {s['adaptive_top1']:.2f}   no_webcam {s['no_webcam_top1']:.2f}")
    print("\nby patient (adaptive top-1 / top-3):")
    for k, s in rep["by_persona"].items():
        print(f"  {k:<10} {s['adaptive_top1']:.2f} / {s['adaptive_top3']:.2f}   (n={s['n']})")
    for k, d in rep["routine_change"].items():
        print(f"\nroutine change for {k} on day {d['change_on_day']} (adaptive top-1):  before "
              f"{d['before (3 days)']['adaptive_top1']}  ->  days 1-3 after {d['days 1-3 after']['adaptive_top1']}"
              f"  ->  days 4-7 after {d['days 4-7 after']['adaptive_top1']}")
    print("\nfinal expert weights (adaptive):")
    for k, w in rep["final_expert_weights"].items():
        print(f"  {k:<10} " + ", ".join(f"{n} {v:.2f}" for n, v in list(w.items())[:4]))
    print(f"\nreport: {REPORT_DIR / 'evaluation.json'}")


# =========================================================================== seed
def seed(days: int, seed_: int, reset: bool) -> None:
    from sqlalchemy import delete, insert, select

    from app import models
    from app.adaptive_service import service
    from app.database import Base, SessionLocal, engine as db_engine
    from adaptive.config import from_epoch

    Base.metadata.create_all(bind=db_engine)
    rng = np.random.default_rng(seed_)
    now = time.time()
    start = local_midnight(days)
    with SessionLocal() as db:
        sims = db.scalars(select(models.Patient).where(models.Patient.notes.like(f"{SIM_TAG}%"))).all()
        if sims and not reset:
            print(f"{len(sims)} simulated patients already exist; use --reset to recreate them.")
            return
        for p in sims:
            pid = p.id
            for M in (models.Request, models.GazeWindow, models.GazeEvent, models.CareFeedback,
                      models.PredictionEval, models.CaptureSession):
                db.execute(delete(M).where(M.patient_id == pid))
            db.execute(delete(models.ModelState).where(models.ModelState.key == f"patient:{pid}"))
            db.delete(p)
        db.execute(delete(models.ModelState).where(models.ModelState.key == "global"))
        db.commit()

        for persona in PERSONAS:
            g = generate(persona, start, days + 1, rng, end_at=now)
            pat = models.Patient(name=persona.name, notes=f"{SIM_TAG}{persona.key}] {persona.notes}")
            db.add(pat)
            db.flush()
            db.execute(insert(models.Request), [
                {"patient_id": pat.id, "ts": from_epoch(t), "request": l, "source": "simulated"}
                for t, l, _ in g["requests"]])
            rows = [{"patient_id": pat.id, "start": from_epoch(t), "simulated": True, "valid": 100,
                     "user_id": f"USER{pat.id:03d}", "time_window": _tw(t), **_cols(v)} for t, v in g["windows"]]
            for i in range(0, len(rows), 5000):
                db.execute(insert(models.GazeWindow), rows[i:i + 5000])
            db.commit()
            print(f"  {pat.id:>3}  {persona.name:<26} {len(g['requests']):>4} requests  {len(rows):>6} windows")
    service.retrain_all()
    print("adaptive state trained for all patients (replayed every request)")


def _cols(v: dict) -> dict:
    from app.models import WINDOW_COLUMNS
    return {col: v[f] for f, col in WINDOW_COLUMNS.items()}


def _tw(t: float) -> str:
    a, b = local_dt(t), local_dt(t + 10)
    return f"{a:%H:%M:%S}-{b:%H:%M:%S}"


# =========================================================================== live
def live(api: str, speed: float, relay: str | None, minutes: float | None) -> None:
    import urllib.request

    def call(method, url, body=None):
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read() or b"null")

    pats = [p for p in call("GET", f"{api}/patients") if (p.get("notes") or "").startswith(SIM_TAG)]
    if not pats:
        sys.exit("No simulated patients found. Run: python -m sim.ward seed")
    rng = np.random.default_rng(int(time.time()))
    virt = []
    for p in pats:
        key = p["notes"][len(SIM_TAG):].split("]")[0]
        persona = BY_KEY.get(key)
        if persona:
            virt.append({"id": p["id"], "persona": persona, "sigs": [], "base": None, "next_need": None})
    from .generator import DEFAULT_BASE
    for v in virt:
        v["base"] = {**DEFAULT_BASE, **v["persona"].baseline}
        v["next_need"] = time.time() + rng.exponential(180 / speed)

    print(f"streaming {len(virt)} virtual patients to {api} (Ctrl+C to stop)")
    stop_at = time.time() + minutes * 60 if minutes else None
    next_win = math.ceil(time.time() / 10) * 10
    while stop_at is None or time.time() < stop_at:
        time.sleep(max(0, next_win + 0.3 - time.time()))
        w = next_win
        next_win += 10
        for v in virt:
            # schedule the next need: label from routine at this hour + episodes, with a build-up
            if v["next_need"] and time.time() >= v["next_need"] - 1:
                label, state = _live_need(v["persona"], rng)
                t_req = time.time() + (rng.uniform(60, 150) if state else 0)
                if state:
                    v["sigs"].append(Signature(time.time(), t_req, state, rng.uniform(0.7, 1.0), True))
                v["pending"] = (t_req, label)
                v["next_need"] = None
            vals = window_values(rng, v["base"], intensity_at(v["sigs"], w + 10))
            pend = v.get("pending")
            fire = pend and pend[0] <= w + 10
            if fire and pend[1] == "help":
                vals["Emergency_Button"] = 1
            env = {"message": {"User_ID": f"USER{v['id']:03d}", "Time_Window": _tw(w), **vals},
                   "key": f"{v['id']}|{int(w * 1000)}", "simulated": True}
            try:
                call("POST", f"{api}/windows", {**env, "valid": 100})
                if relay:
                    call("POST", f"{relay}/window", env)
            except Exception as e:  # keep streaming through a backend restart
                print("  window post failed:", e)
            if fire:
                ts = datetime.fromtimestamp(pend[0], timezone.utc).isoformat().replace("+00:00", "Z")
                try:
                    call("POST", f"{api}/requests", {"patient_id": v["id"], "ts": ts, "request": pend[1], "source": "simulated"})
                    print(f"  {local_dt(pend[0]):%H:%M:%S}  {v['persona'].name:<26} -> {pend[1]}")
                except Exception as e:
                    print("  request post failed:", e)
                v["pending"] = None
                v["sigs"] = [s for s in v["sigs"] if s.end > time.time() - 60]
                v["next_need"] = time.time() + rng.exponential(240 / speed)


def _live_need(persona, rng):
    """Pick the next need for live mode: an episode (with webcam build-up) or the routine at this hour."""
    now = local_dt(time.time())
    m = now.hour * 60 + now.minute
    eps = [e for e in persona.episodes if e.hours[0] * 60 <= m <= e.hours[1] * 60] or persona.episodes
    if eps and rng.random() < 0.45:
        e = eps[int(rng.integers(len(eps)))]
        labels, p = zip(*e.labels.items())
        return labels[int(rng.choice(len(labels), p=np.array(p) / sum(p)))], e.state
    if persona.routines:
        d = np.array([min(abs(r.minute - m), 1440 - abs(r.minute - m)) for r in persona.routines], dtype=float)
        w = np.exp(-0.5 * (d / 60) ** 2) + 0.02
        r = persona.routines[int(rng.choice(len(w), p=w / w.sum()))]
        return r.label, None
    return LABELS[int(rng.integers(len(LABELS)))], None


# =========================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="offline learning-curve evaluation (no database)")
    e.add_argument("--days", type=int, default=21)
    e.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("seed", help="write virtual patients into the database")
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--reset", action="store_true", help="delete and recreate existing simulated patients")
    lv = sub.add_parser("live", help="stream live windows and requests to the backend")
    lv.add_argument("--api", default="http://localhost:8000")
    lv.add_argument("--speed", type=float, default=1.0, help="higher = needs come up more often")
    lv.add_argument("--relay", default=None, help="also post windows to the B1 server (held as simulated)")
    lv.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    a = ap.parse_args(argv)
    if a.cmd == "evaluate":
        evaluate(a.days, a.seed)
    elif a.cmd == "seed":
        seed(a.days, a.seed, a.reset)
    else:
        try:
            live(a.api, a.speed, a.relay, a.minutes)
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
