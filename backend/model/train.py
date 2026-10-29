"""Train the request-suggestion model and tune the confusion threshold.

    python -m model.train                 # run from the backend/ directory
    python -m model.train --refit-all     # evaluate on a time split, then refit on all data

- One RandomForest per patient (routines differ, and patient_id is not a feature),
  plus a global RandomForest used as a fallback for unknown patients.
- Rows: one per request, features computed at (request ts - U(5, 60) s) using only
  data from before that moment, label = the request that follows.
- Time-based split: the last --test-frac of days is the test set.
- Confusion: gaze entropy (normalised, 0-1) over the last 30 s; the threshold is tuned
  on the training period (maximises TPR - FPR) against simulator ground truth in
  episodes.csv, then reported on the test period.
- Saved with joblib to model/artifacts/model.joblib.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import (
    FEATURE_NAMES, LABEL_IDX, LABELS, MIN_EVENTS, PRIOR_SIGMA_MIN, WINDOW_S,
    build_features, gaze_window_stats, minute_of_day, time_prior, to_ns, zone_codes,
)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
ARTIFACT_DIR = HERE / "artifacts"


# ------------------------------------------------------------------ data
def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[ns]")


def load(data_dir: Path):
    req = pd.read_csv(data_dir / "requests.csv")
    gaze = pd.read_csv(data_dir / "gaze.csv")
    ep = pd.read_csv(data_dir / "episodes.csv")
    req["ts"], gaze["ts"] = _ts(req["ts"]), _ts(gaze["ts"])
    ep["start"], ep["end"] = _ts(ep["start"]), _ts(ep["end"])
    return req.sort_values(["patient_id", "ts"]), gaze.sort_values(["patient_id", "ts"]), ep


def patient_arrays(req, gaze, pid):
    r, g = req[req.patient_id == pid], gaze[gaze.patient_id == pid]
    return {
        "req_ts": r.ts.to_numpy("datetime64[ns]"),
        "req_codes": np.array([LABEL_IDX[l] for l in r.request], dtype=np.int64),
        "g_ts": g.ts.to_numpy("datetime64[ns]"),
        "g_zone": zone_codes(g.zone),
        "g_dwell": g.dwell_ms.to_numpy(float),
    }


def feats(a, ts):
    return build_features(ts, a["req_ts"], a["req_codes"], a["g_ts"], a["g_zone"], a["g_dwell"])


def make_rows(a, rng):
    """One row per request: features shortly before it (lead 5-60 s), label = the request."""
    X, y, t_req, t_query = [], [], [], []
    for t, c in zip(a["req_ts"], a["req_codes"]):
        tq = t - np.timedelta64(int(rng.uniform(5, 60) * 1000), "ms")
        x, _ = feats(a, tq)
        X.append(x); y.append(c); t_req.append(t); t_query.append(tq)
    return {"X": np.array(X), "y": np.array(y), "t_req": np.array(t_req), "t_query": np.array(t_query)}


def subset(rows, mask):
    return {k: v[mask] for k, v in rows.items()}


# ----------------------------------------------------------------- model
def fit_rf(X, y, trees, seed):
    return RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, n_jobs=-1, random_state=seed
    ).fit(X, y)


def full_proba(model, X):
    """predict_proba mapped onto all 6 labels (a model may not have seen every label)."""
    out = np.zeros((len(X), len(LABELS)))
    out[:, model.classes_.astype(int)] = model.predict_proba(X)
    return out


def topk_acc(P, y, k):
    top = np.argsort(-P, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def prior_minutes(req_df):
    out = {}
    for c, l in enumerate(LABELS):
        ts = req_df.loc[req_df.request == l, "ts"]
        out[c] = (ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60).to_numpy(float)
    return out


# ------------------------------------------------------------- confusion
def score_windows(ep, arrays_by_pid, rng, per_episode=8):
    """Entropy at random points inside simulator episodes (window fully inside the episode)."""
    rows = []
    for pid, start, end, kind in ep[["patient_id", "start", "end", "kind"]].itertuples(index=False):
        a = arrays_by_pid[pid]
        lo, hi = start + pd.Timedelta(seconds=WINDOW_S), end
        if hi <= lo:
            continue
        span_ms = (hi - lo) / pd.Timedelta(milliseconds=1)
        for off in rng.uniform(0, span_ms, per_episode):
            t = lo + pd.Timedelta(milliseconds=float(off))
            _, ent, _, n = gaze_window_stats(a["g_ts"], a["g_zone"], a["g_dwell"], t)
            rows.append((pid, t, ent, n, kind == "confused"))
    return pd.DataFrame(rows, columns=["patient_id", "t", "entropy", "n", "confused"])


def tune_threshold(w):
    """Threshold on entropy maximising TPR - FPR; ties -> middle of the tied range."""
    w = w[w.n >= MIN_EVENTS]
    pos, neg = w.entropy[w.confused].to_numpy(), w.entropy[~w.confused].to_numpy()
    s = np.unique(w.entropy.to_numpy())
    cands = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    J = np.array([(pos > c).mean() - (neg > c).mean() for c in cands])
    best = cands[J >= J.max() - 1e-12]
    return float(np.median(best)), float(J.max())


def confusion_metrics(w, thr):
    w = w[w.n >= MIN_EVENTS]
    flag = w.entropy > thr
    pos, neg = w[w.confused], w[~w.confused]
    return {
        "threshold": round(thr, 4),
        "confused_windows": int(len(pos)), "calm_windows": int(len(neg)),
        "recall_confused": round(float(flag[w.confused].mean()), 4) if len(pos) else None,
        "false_alarm_rate": round(float(flag[~w.confused].mean()), 4) if len(neg) else None,
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "model.joblib")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit-all", action="store_true",
                    help="after evaluating on the time split, refit the saved model on all data")
    args = ap.parse_args()

    req, gaze, ep = load(args.data_dir)
    pids = sorted(req.patient_id.unique().tolist())
    arrays = {p: patient_arrays(req, gaze, p) for p in pids}

    first = req.ts.min().normalize()
    n_days = (req.ts.max().normalize() - first).days + 1
    cutoff = first + pd.Timedelta(days=round(n_days * (1 - args.test_frac)))
    cut64 = to_ns(cutoff)
    print(f"{len(req)} requests over {n_days} days; train < {cutoff.date()} <= test")

    rng = np.random.default_rng(args.seed)
    rows = {p: make_rows(arrays[p], rng) for p in pids}
    train = {p: subset(rows[p], rows[p]["t_req"] < cut64) for p in pids}
    test = {p: subset(rows[p], rows[p]["t_req"] >= cut64) for p in pids}

    # --- classifier: per patient + global fallback ---
    models = {p: fit_rf(train[p]["X"], train[p]["y"], args.trees, args.seed) for p in pids}
    Xtr = np.vstack([train[p]["X"] for p in pids]); ytr = np.concatenate([train[p]["y"] for p in pids])
    global_model = fit_rf(Xtr, ytr, args.trees, args.seed)

    train_req = req[req.ts < cutoff]
    prior_by_pid = {p: prior_minutes(train_req[train_req.patient_id == p]) for p in pids}

    report = {"per_patient": {}, "cutoff": str(cutoff.date())}
    pooled = {"model": ([], []), "prior": ([], []), "global": ([], [])}
    print(f"\n{'patient':>8} {'n_test':>6} | {'model top1':>10} {'top3':>6} | {'prior top1':>10} {'top3':>6}")
    for p in pids:
        te = test[p]
        if len(te["y"]) == 0:
            continue
        P_model = full_proba(models[p], te["X"])
        P_global = full_proba(global_model, te["X"])
        P_prior = np.array([time_prior(prior_by_pid[p], minute_of_day(t), PRIOR_SIGMA_MIN) for t in te["t_query"]])
        for name, P in (("model", P_model), ("prior", P_prior), ("global", P_global)):
            pooled[name][0].append(P); pooled[name][1].append(te["y"])
        report["per_patient"][p] = {
            "n_test": int(len(te["y"])),
            "model_top1": topk_acc(P_model, te["y"], 1), "model_top3": topk_acc(P_model, te["y"], 3),
            "prior_top1": topk_acc(P_prior, te["y"], 1), "prior_top3": topk_acc(P_prior, te["y"], 3),
        }
        r = report["per_patient"][p]
        print(f"{p:>8} {r['n_test']:>6} | {r['model_top1']:>10.3f} {r['model_top3']:>6.3f} | "
              f"{r['prior_top1']:>10.3f} {r['prior_top3']:>6.3f}")
    overall = {}
    for name, (Ps, ys) in pooled.items():
        P, y = np.vstack(Ps), np.concatenate(ys)
        overall[name] = {"n": int(len(y)), "top1": topk_acc(P, y, 1), "top3": topk_acc(P, y, 3)}
    report["overall"] = overall
    print(f"\n{'overall':>8} {overall['model']['n']:>6} | {overall['model']['top1']:>10.3f} {overall['model']['top3']:>6.3f} | "
          f"{overall['prior']['top1']:>10.3f} {overall['prior']['top3']:>6.3f}   (per-patient RF vs time-of-day prior)")
    print(f"         global RF without patient id: top1 {overall['global']['top1']:.3f}  top3 {overall['global']['top3']:.3f}")

    # --- confusion threshold ---
    W = score_windows(ep, arrays, np.random.default_rng(args.seed + 1))
    W_train, W_test = W[W.t < cutoff], W[W.t >= cutoff]
    thr, J = tune_threshold(W_train)
    report["confusion"] = {"train_J": round(J, 4), "train": confusion_metrics(W_train, thr),
                           "test": confusion_metrics(W_test, thr)}
    # false alarms at the moments the model is actually used: just before real requests
    q_flag = []
    for p in pids:
        for tq in test[p]["t_query"]:
            _, ent, _, n = gaze_window_stats(arrays[p]["g_ts"], arrays[p]["g_zone"], arrays[p]["g_dwell"], tq)
            q_flag.append(n >= MIN_EVENTS and ent > thr)
    report["confusion"]["test_false_alarm_on_request_points"] = round(float(np.mean(q_flag)), 4)
    print("\nconfusion (entropy > threshold):")
    print(f"  tuned threshold {thr:.3f} (train TPR-FPR = {J:.3f})")
    for k in ("train", "test"):
        m = report["confusion"][k]
        print(f"  {k:>5}: recall {m['recall_confused']}  false-alarm {m['false_alarm_rate']}  "
              f"({m['confused_windows']} confused / {m['calm_windows']} calm windows)")
    print(f"  false alarms at test request points: {report['confusion']['test_false_alarm_on_request_points']}")

    # --- final bundle ---
    if args.refit_all:
        print("\nrefitting on all data for the saved model")
        models = {p: fit_rf(rows[p]["X"], rows[p]["y"], args.trees, args.seed) for p in pids}
        global_model = fit_rf(np.vstack([rows[p]["X"] for p in pids]),
                              np.concatenate([rows[p]["y"] for p in pids]), args.trees, args.seed)
        prior_by_pid = {p: prior_minutes(req[req.patient_id == p]) for p in pids}
        thr, _ = tune_threshold(W)
        train_req = req
    bundle = {
        "labels": LABELS, "feature_names": FEATURE_NAMES,
        "models": models, "global_model": global_model,
        "prior": prior_by_pid, "prior_global": prior_minutes(train_req),
        "prior_sigma_min": PRIOR_SIGMA_MIN,
        "threshold": thr, "min_events": MIN_EVENTS, "window_s": WINDOW_S,
        "refit_all": args.refit_all, "metrics": report,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    (args.out.parent / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved {args.out}  (metrics -> {args.out.parent / 'metrics.json'})")


if __name__ == "__main__":
    main()
