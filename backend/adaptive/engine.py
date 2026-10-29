"""Adaptive ensemble: experts -> Hedge-weighted mixture -> needs, state, caregiver actions.

Learning is prequential: when a request arrives at time t, the engine first predicts as of
t - 15 s (using only what was known then), scores every expert on the real answer, then
updates. So every stored evaluation is an honest out-of-sample test, and the running
accuracy IS the learning curve.

    weights   Hedge with specialists (sleeping experts keep their weight) + fixed share,
              so the mix re-adapts when a patient's routine changes
    learned   per-patient online softmax + a global one shared by all patients
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from .config import (
    EXPERT_SMOOTH, ETA, FEEDBACK_DISMISS_S, FEEDBACK_DONE_S, FEEDBACK_FACTOR, FIXED_SHARE,
    GAZE_FOCUS_S, LABEL_TEXT, LABELS, LEAD_S, N, PATIENT_TZ, PRIOR_WEIGHTS, fmt_duration,
)
from .data import FEEDBACK_NAMES, NONE_ZONE, PatientData, local_parts
from . import experts as ex
from . import suggest
from . import webcam as wc
from .online import LogPool, pool_inputs

ENGINE_VERSION = 3
CONFUSION_THRESHOLD = 0.8585   # tuned by model/train.py on simulator ground truth
CONFUSION_MIN_EVENTS = 5
ROLLING = 50

EXPERT_TEXT = {
    "routine": "Daily routine", "transition": "What follows the last request", "due": "Overdue needs",
    "webcam": "Webcam state", "gaze_focus": "Current gaze focus", "learned": "Personal learned model",
    "learned_global": "Ward-wide learned model", "static_rf": "Offline RandomForest",
}


def gaze_entropy(h: PatientData, t: float):
    """Dwell-weighted gaze entropy over the last 30 s (same definition as model/features.py)."""
    m = (h.gaze_t > t - GAZE_FOCUS_S) & (h.gaze_t <= t)
    n = int(m.sum())
    if n == 0:
        return 0.0, 0
    z = np.where(h.gaze_z[m] == NONE_ZONE, N, h.gaze_z[m])
    w = np.bincount(z, weights=h.gaze_d[m], minlength=N + 1)
    if w.sum() <= 0:
        w = np.bincount(z, minlength=N + 1).astype(float)
    p = w[w > 0] / w.sum()
    return float(-(p * np.log(p)).sum() / math.log(N + 1)), n


class Engine:
    def __init__(self, use_webcam: bool = True, static_rf=None, tz=PATIENT_TZ, disabled=()):
        self.use_webcam = use_webcam
        self.static_rf = static_rf
        self.tz = tz
        self.disabled = set(disabled) | (set() if use_webcam else {"webcam"})

    # ----------------------------------------------------------------- state
    def new_state(self) -> dict:
        return {
            "version": ENGINE_VERSION, "weights": dict(PRIOR_WEIGHTS), "learned": None,
            "n": 0, "last_t": None, "recent": [], "hits": {"top1": 0, "top3": 0},
            "expert_hits": {k: 0.0 for k in PRIOR_WEIGHTS}, "expert_seen": {k: 0 for k in PRIOR_WEIGHTS},
        }

    @staticmethod
    def new_global_state() -> dict:
        return {"version": ENGINE_VERSION, "learned": None}

    # ---------------------------------------------------------------- core
    def _context(self, h: PatientData, t: float) -> dict:
        mod, wd = local_parts(np.array([t]), self.tz)
        ws = wc.assess(h, t) if self.use_webcam else {"available": False, "last_window_age_s": None}
        return {"mod": float(mod[0]), "wd": int(wd[0]), "webcam": ws}

    def _experts(self, h, t, ctx, state, gstate, patient_id):
        out = {}
        for name, fn in (("routine", ex.routine), ("transition", ex.transition), ("due", ex.due),
                         ("webcam", ex.webcam), ("gaze_focus", ex.gaze_focus)):
            if name not in self.disabled:
                out[name] = fn(h, t, ctx)
        if self.static_rf is not None and "static_rf" not in self.disabled:
            out["static_rf"] = self.static_rf(h, t, ctx, patient_id)
        x = pool_inputs(out, ctx["webcam"])
        g_learned = (gstate or {}).get("learned")
        mine = LogPool(state.get("learned"), init=g_learned)
        glob = LogPool(g_learned)
        if "learned" not in self.disabled:
            out["learned"] = (mine.proba(*x), []) if mine.n >= 3 else (None, [])
        if gstate is not None and "learned_global" not in self.disabled:
            out["learned_global"] = (glob.proba(*x), []) if glob.n >= 10 else (None, [])
        return out, x, mine, glob

    @staticmethod
    def _mix(outs: dict, weights: dict):
        awake = {k: (1 - EXPERT_SMOOTH) * p + EXPERT_SMOOTH / N for k, (p, _) in outs.items() if p is not None}
        if not awake:
            return np.full(N, 1.0 / N), awake, {}
        tot = sum(weights.get(k, 0.5) for k in awake)
        share = {k: weights.get(k, 0.5) / tot for k in awake}
        p = sum(share[k] * awake[k] for k in awake)
        return p / p.sum(), awake, share

    # -------------------------------------------------------------- predict
    def predict(self, data: PatientData, state: dict, t: float, gstate: dict | None = None,
                patient_id=None, with_actions: bool = True) -> dict:
        h = data.view(t + 1e-6)  # everything up to and including "now"
        ctx = self._context(h, t)
        outs, _, mine, _ = self._experts(h, t, ctx, state, gstate, patient_id)
        p, awake, share = self._mix(outs, state["weights"])
        p_model = p.copy()

        # caregiver feedback: recently handled or dismissed needs are less likely right now
        applied = []
        for i in range(len(h.fb_t)):
            age = t - h.fb_t[i]
            action = FEEDBACK_NAMES[int(h.fb_a[i])]
            limit = FEEDBACK_DONE_S if action == "done" else FEEDBACK_DISMISS_S
            if 0 <= age <= limit:
                c = int(h.fb_y[i])
                p[c] *= FEEDBACK_FACTOR[action]
                applied.append({"label": LABELS[c], "action": action, "age_s": round(age)})
        p = p / p.sum()

        reasons = {l: [] for l in LABELS}
        for k, (pk, rs) in outs.items():
            if pk is None:
                continue
            for r in rs:
                if r["label"]:
                    reasons[r["label"]].append(r["text"])
        ws = ctx["webcam"]
        if ws.get("available"):
            for k, s in ws["states"].items():
                if s >= 0.5:
                    for lbl in LABELS:
                        if ex.STATE_NEEDS[k][LABELS.index(lbl)] >= 0.3:
                            reasons[lbl].append(f"Webcam: {ex.STATE_TEXT[k]} ({wc.describe(ws, k)})")
        for f in applied:
            reasons[f["label"]].append(
                f"Caregiver marked {'done' if f['action'] == 'done' else 'not needed'} {fmt_duration(f['age_s'])} ago")

        order = np.argsort(-p)
        needs = [{"label": LABELS[i], "prob": round(float(p[i]), 4), "reasons": reasons[LABELS[i]]} for i in order]

        ent, n_ev = gaze_entropy(h, t)
        confused_gaze = n_ev >= CONFUSION_MIN_EVENTS and ent > CONFUSION_THRESHOLD
        confused_cam = bool(ws.get("available") and ws["states"]["confusion"] >= 0.75)
        confusion_score = max(ent, ws["states"]["confusion"] if ws.get("available") else 0.0)

        expert_rows = []
        used = [k for k in PRIOR_WEIGHTS if k in outs and (state["expert_seen"].get(k) or outs[k][0] is not None)]
        wsum = sum(state["weights"].get(k, 0) for k in used) or 1.0
        for k in PRIOR_WEIGHTS:
            if k not in outs:
                continue
            pk = outs[k][0]
            expert_rows.append({
                "name": k, "text": EXPERT_TEXT[k],
                "weight": round(state["weights"].get(k, 0) / wsum, 4) if k in used else None,
                "scored": state["expert_seen"].get(k, 0),
                "awake": pk is not None, "share": round(share.get(k, 0.0), 4),
                "top": LABELS[int(np.argmax(pk))] if pk is not None else None,
                "top_prob": round(float(np.max(pk)), 4) if pk is not None else None,
                "accuracy": round(state["expert_hits"].get(k, 0) / state["expert_seen"][k], 4)
                if state["expert_seen"].get(k) else None,
            })

        recent = state.get("recent", [])
        out = {
            "ts": t,
            "needs": needs,
            "suggestions": [{"label": n["label"], "prob": n["prob"]} for n in needs[:3]],
            "confused": bool(confused_gaze or confused_cam),
            "confusion_score": round(float(confusion_score), 4),
            "gaze_entropy": round(ent, 4),
            "webcam": ws,
            "feedback_applied": applied,
            "experts": expert_rows,
            "model": {
                "engine_version": ENGINE_VERSION,
                "n_learned": state["n"],
                "personal_model_updates": mine.n,
                "personal_model": mine.explain() if mine.n else None,
                "rolling_window": len(recent),
                "rolling_top1": round(sum(r[0] for r in recent) / len(recent), 4) if recent else None,
                "rolling_top3": round(sum(r[1] for r in recent) / len(recent), 4) if recent else None,
                "overall_top1": round(state["hits"]["top1"] / state["n"], 4) if state["n"] else None,
                "overall_top3": round(state["hits"]["top3"] / state["n"], 4) if state["n"] else None,
                "uncalibrated_top": LABELS[int(np.argmax(p_model))],
            },
        }
        if with_actions:
            last_req = None
            fb_after = False
            if h.n_requests:
                lt, ly = float(h.req_t[-1]), int(h.req_y[-1])
                last_req = {"t": lt, "label": LABELS[ly]}
                fb_after = bool(((h.fb_t >= lt) & (h.fb_y == ly)).any())
            up = ex.upcoming(h, t, ctx)
            alert, actions = suggest.build(needs, ws, out["confused"], last_req, fb_after, t, up)
            out.update(alert=alert, actions=actions, upcoming=up)
            if last_req:
                out["last_request"] = {**last_req, "age_s": round(t - last_req["t"])}
        return out

    # ---------------------------------------------------------------- learn
    def learn(self, data: PatientData, state: dict, t: float, label: str, gstate: dict | None = None,
              patient_id=None) -> dict:
        """Score the prediction made at t - LEAD_S against `label`, then update. Mutates state."""
        y = LABELS.index(label)
        q = t - LEAD_S
        h = data.view(q)
        ctx = self._context(h, q)
        outs, x, mine, glob = self._experts(h, q, ctx, state, gstate, patient_id)
        p, awake, share = self._mix(outs, state["weights"])
        rank = int((p >= p[y] - 1e-12).sum())  # ties count against the model
        rec = {
            "t": t, "label": label, "p_actual": round(float(p[y]), 4), "rank": rank,
            "top1": rank == 1, "top3": rank <= 3, "predicted": LABELS[int(np.argmax(p))],
            "experts": {k: round(float(v[y]), 4) for k, v in awake.items()},
            "webcam_available": bool(ctx["webcam"].get("available")),
        }

        # Hedge for specialists: only awake experts are scored, and their total weight is
        # preserved, so an expert is never punished for staying silent.
        w = state["weights"]
        if awake:
            before = sum(w.get(k, 0.5) for k in awake)
            for k, pk in awake.items():
                w[k] = w.get(k, 0.5) * float(pk[y]) ** ETA
                state["expert_seen"][k] = state["expert_seen"].get(k, 0) + 1
                state["expert_hits"][k] = state["expert_hits"].get(k, 0) + (1.0 if int(np.argmax(pk)) == y else 0.0)
            after = sum(w[k] for k in awake) or 1e-12
            for k in awake:
                w[k] *= before / after
            mean = sum(w.values()) / len(w)
            for k in w:
                w[k] = max(1e-4, (1 - FIXED_SHARE) * w[k] + FIXED_SHARE * mean)

        mine.update(*x, y)
        state["learned"] = mine.state()
        if gstate is not None:
            glob.update(*x, y)
            gstate["learned"] = glob.state()

        state["n"] += 1
        state["last_t"] = t
        state["hits"]["top1"] += int(rec["top1"])
        state["hits"]["top3"] += int(rec["top3"])
        state["recent"] = (state.get("recent", []) + [[int(rec["top1"]), int(rec["top3"])]])[-ROLLING:]
        return rec

    def snapshot(self, state: dict) -> dict:
        return deepcopy(state)


def describe_need(label: str) -> str:
    return LABEL_TEXT[label]
