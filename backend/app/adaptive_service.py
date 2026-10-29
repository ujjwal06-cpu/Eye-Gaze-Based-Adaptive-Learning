"""Glue between the database and the adaptive engine.

- loads a patient's history (requests, webcam windows, recent gaze, caregiver feedback)
- keeps the adaptive state in the model_state table
- learns from every new request (prequential: predict first, then update) and stores the
  evaluation in prediction_evals, so accuracy over time is always available
- if requests were added some other way (seed scripts, CSV import, an older backend), the
  next call catches up by replaying them in time order
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from adaptive import ENGINE_VERSION, Engine, PatientData
from adaptive.config import BASELINE_LOOKBACK_S, GAZE_FOCUS_S, HISTORY_DAYS, LEAD_S, from_epoch, to_epoch
from adaptive.rf import StaticRF

from . import models
from .database import SessionLocal

log = logging.getLogger("adaptive")
LEARN_SOURCES = {s.strip() for s in os.getenv("ADAPTIVE_LEARN_SOURCES", "gaze,simulated").split(",") if s.strip()}
USE_STATIC_RF = os.getenv("ADAPTIVE_STATIC_RF", "1") != "0"
COL_TO_FIELD = {c: f for f, c in models.WINDOW_COLUMNS.items()}


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _isoify(o):
    """Epoch-second fields named t / ts / at -> ISO-8601 UTC strings, recursively."""
    if isinstance(o, dict):
        return {k: (iso(v) if k in ("t", "ts", "at") and isinstance(v, (int, float)) else _isoify(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_isoify(v) for v in o]
    return o


def window_message(w: models.GazeWindow) -> dict:
    """Stored row -> the exact B1 message that was sent."""
    msg = {"User_ID": w.user_id, "Time_Window": w.time_window}
    for f, c in models.WINDOW_COLUMNS.items():
        msg[f] = getattr(w, c)
    return msg


class AdaptiveService:
    def __init__(self):
        self.engine = Engine(static_rf=StaticRF() if USE_STATIC_RF else None)
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- state
    def _get_state(self, db: Session, key: str, fresh):
        row = db.get(models.ModelState, key)
        if row and isinstance(row.data, dict) and row.data.get("version") == ENGINE_VERSION:
            return dict(row.data)
        return fresh()

    def _put_state(self, db: Session, key: str, data: dict) -> None:
        row = db.get(models.ModelState, key)
        if row is None:
            db.add(models.ModelState(key=key, data=data))
        else:
            row.data = dict(data)  # new object so SQLAlchemy sees the change

    # ----------------------------------------------------------------- load
    def load(self, db: Session, pid: int, until: float, req_from: float | None = None,
             win_from: float | None = None, gaze_ranges=None) -> PatientData:
        req_from = req_from if req_from is not None else until - HISTORY_DAYS * 86400
        win_from = win_from if win_from is not None else until - BASELINE_LOOKBACK_S - 300
        reqs = db.execute(
            select(models.Request.ts, models.Request.request)
            .where(models.Request.patient_id == pid, models.Request.source.in_(LEARN_SOURCES),
                   models.Request.ts >= from_epoch(req_from), models.Request.ts <= from_epoch(until))
        ).all()
        cols = [getattr(models.GazeWindow, c) for c in models.WINDOW_COLUMNS.values()]
        wins = db.execute(
            select(models.GazeWindow.start, *cols)
            .where(models.GazeWindow.patient_id == pid,
                   models.GazeWindow.start >= from_epoch(win_from), models.GazeWindow.start <= from_epoch(until))
        ).all()
        gaze = []
        for a, b in gaze_ranges or [(until - GAZE_FOCUS_S - 5, until)]:
            gaze += db.execute(
                select(models.GazeEvent.ts, models.GazeEvent.zone, models.GazeEvent.dwell_ms)
                .where(models.GazeEvent.patient_id == pid,
                       models.GazeEvent.ts > from_epoch(a), models.GazeEvent.ts <= from_epoch(b))
            ).all()
        fb = db.execute(
            select(models.CareFeedback.ts, models.CareFeedback.label, models.CareFeedback.action)
            .where(models.CareFeedback.patient_id == pid, models.CareFeedback.ts >= from_epoch(until - 6 * 3600),
                   models.CareFeedback.ts <= from_epoch(until))
        ).all()
        return PatientData.from_records(
            requests=[(to_epoch(t), l) for t, l in reqs],
            windows=[(to_epoch(r[0]), list(r[1:])) for r in wins],
            gaze=[(to_epoch(t), z, d) for t, z, d in gaze],
            feedback=[(to_epoch(t), l, a) for t, l, a in fb],
        )

    # ---------------------------------------------------------------- learn
    def catch_up(self, db: Session, pid: int) -> int:
        """Learn every request of this patient not learned yet, in time order. Returns how many."""
        with self.lock:
            key = f"patient:{pid}"
            state = self._get_state(db, key, self.engine.new_state)
            gstate = self._get_state(db, "global", Engine.new_global_state)
            last_id = state.get("last_request_id") or 0
            todo = db.execute(
                select(models.Request.id, models.Request.ts, models.Request.request, models.Request.source)
                .where(models.Request.patient_id == pid, models.Request.id > last_id,
                       models.Request.source.in_(LEARN_SOURCES))
                .order_by(models.Request.ts, models.Request.id)
            ).all()
            if not todo:
                return 0
            ts = [to_epoch(r.ts) for r in todo]
            first, last = min(ts), max(ts)
            ranges = [(t - LEAD_S - GAZE_FOCUS_S - 5, t) for t in ts] if len(ts) <= 2000 else None
            data = self.load(db, pid, until=last + 1, req_from=first - HISTORY_DAYS * 86400,
                             win_from=first - BASELINE_LOOKBACK_S - 300, gaze_ranges=ranges)
            rows = []
            for r, t in zip(todo, ts):
                rec = self.engine.learn(data, state, t, r.request, gstate, patient_id=pid)
                rows.append(models.PredictionEval(
                    patient_id=pid, request_id=r.id, ts=from_epoch(t), actual=rec["label"], predicted=rec["predicted"],
                    p_actual=rec["p_actual"], rank=rec["rank"], top1=rec["top1"], top3=rec["top3"],
                    experts=rec["experts"], webcam_available=rec["webcam_available"], source=r.source,
                    engine_version=ENGINE_VERSION))
            state["last_request_id"] = max(last_id, max(r.id for r in todo))
            db.add_all(rows)
            self._put_state(db, key, state)
            self._put_state(db, "global", gstate)
            db.commit()
            return len(todo)

    def catch_up_all(self) -> None:
        """Background start-up task: learn anything added while the server was down."""
        try:
            with SessionLocal() as db:
                for pid in db.scalars(select(models.Patient.id).order_by(models.Patient.id)).all():
                    n = self.catch_up(db, pid)
                    if n:
                        log.info("adaptive: learned %d requests for patient %s", n, pid)
        except Exception:
            log.exception("adaptive catch-up at start-up failed")

    def safe_catch_up(self, pid: int) -> None:
        """Used after POST /requests: learning must never make the request itself fail."""
        try:
            with SessionLocal() as db:
                self.catch_up(db, pid)
        except Exception:
            log.exception("adaptive learning failed for patient %s", pid)

    def retrain(self, db: Session, pid: int) -> int:
        with self.lock:
            db.execute(delete(models.PredictionEval).where(models.PredictionEval.patient_id == pid))
            db.execute(delete(models.ModelState).where(models.ModelState.key == f"patient:{pid}"))
            db.commit()
            return self.catch_up(db, pid)

    def retrain_all(self) -> dict:
        with self.lock, SessionLocal() as db:
            db.execute(delete(models.PredictionEval))
            db.execute(delete(models.ModelState))
            db.commit()
            # caregiver feedback is not replayed: it is read from the database at prediction time
            return {pid: self.catch_up(db, pid) for pid in db.scalars(select(models.Patient.id).order_by(models.Patient.id))}

    # --------------------------------------------------------------- predict
    def insights(self, db: Session, pid: int, t: float | None = None) -> dict:
        now = t if t is not None else datetime.now(timezone.utc).timestamp()
        self.catch_up(db, pid)
        state = self._get_state(db, f"patient:{pid}", self.engine.new_state)
        gstate = self._get_state(db, "global", Engine.new_global_state)
        data = self.load(db, pid, until=now)
        out = self.engine.predict(data, state, now, gstate, patient_id=pid)
        out["patient_id"] = pid
        out["generated_at"] = iso(datetime.now(timezone.utc).timestamp())
        return _isoify(out)


service = AdaptiveService()

__all__ = ["service", "iso", "window_message", "COL_TO_FIELD"]
