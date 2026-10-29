import csv
import io
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from model.predict import predict, set_data_source

from . import models, schemas
from .adaptive_service import iso, service, window_message
from .database import Base, engine, get_db
from .db_source import DBSource

AUTO_CREATE_PATIENTS = os.getenv("AUTO_CREATE_PATIENTS", "1") != "0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    set_data_source(DBSource())  # predict() reads history from this database
    if os.getenv("ADAPTIVE_CATCH_UP_ON_START", "1") != "0":
        threading.Thread(target=service.catch_up_all, name="adaptive-catch-up", daemon=True).start()
    yield


app = FastAPI(title="Gaze Request API", version="2.0.0", lifespan=lifespan)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wall(dt: datetime) -> datetime:
    """Timestamps are the patient's wall-clock time: drop any tz without converting
    (same convention as the model)."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _patient_or_404(db: Session, patient_id: int) -> None:
    if db.get(models.Patient, patient_id) is None:
        raise HTTPException(404, f"Patient {patient_id} not found")


def _ensure_patients(db: Session, ids) -> None:
    """Unknown patient ids from GazePad are registered on first contact (AUTO_CREATE_PATIENTS=0 to 404)."""
    ids = {int(i) for i in ids}
    found = set(db.scalars(select(models.Patient.id).where(models.Patient.id.in_(ids))).all())
    missing = ids - found
    if not missing:
        return
    if not AUTO_CREATE_PATIENTS:
        raise HTTPException(404, f"Unknown patient ids: {sorted(missing)}")
    for i in sorted(missing):
        db.add(models.Patient(id=i, name=f"Patient {i}", notes="registered automatically by GazePad"))
    try:
        db.commit()
    except IntegrityError:  # created concurrently by another request
        db.rollback()


def _pid(value: str | int) -> int:
    """Query-string patient id: 1, "1", "p1" and "USER001" all mean patient 1."""
    v = schemas.coerce_patient_id(value)
    try:
        return int(v)
    except (TypeError, ValueError):
        raise HTTPException(422, f"invalid patient_id {value!r}")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- patients
@app.get("/patients", response_model=list[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.scalars(select(models.Patient).order_by(models.Patient.id)).all()


# -------------------------------------------------------------------- gaze
@app.post("/gaze", status_code=201)
def post_gaze(body: list[schemas.GazeEventIn] | schemas.GazeBatchIn, db: Session = Depends(get_db)):
    """Batch of gaze events: a JSON array, or {"events": [...]}."""
    events = body.events if isinstance(body, schemas.GazeBatchIn) else body
    if not events:
        return {"inserted": 0}
    _ensure_patients(db, {e.patient_id for e in events})
    db.add_all(
        models.GazeEvent(
            patient_id=e.patient_id, ts=_wall(e.ts), x=e.x, y=e.y,
            zone=e.zone, dwell_ms=e.dwell_ms, confidence=e.confidence,
        )
        for e in events
    )
    db.commit()
    return {"inserted": len(events)}


@app.get("/gaze/recent", response_model=list[schemas.GazeEventOut])
def recent_gaze(
    patient_id: int,
    seconds: int = Query(30, ge=1, le=3600),
    until: datetime | None = Query(None, description="window end; defaults to server local time now"),
    db: Session = Depends(get_db),
):
    _patient_or_404(db, patient_id)
    end = _wall(until) if until else datetime.now()
    stmt = (
        select(models.GazeEvent)
        .where(
            models.GazeEvent.patient_id == patient_id,
            models.GazeEvent.ts > end - timedelta(seconds=seconds),
            models.GazeEvent.ts <= end,
        )
        .order_by(models.GazeEvent.ts)
    )
    return db.scalars(stmt).all()


# ---------------------------------------------------------------- requests
@app.post("/requests", response_model=schemas.RequestOut, status_code=201)
def post_request(body: schemas.RequestIn, background: BackgroundTasks, db: Session = Depends(get_db)):
    _ensure_patients(db, {body.patient_id})
    req = models.Request(
        patient_id=body.patient_id, ts=_wall(body.ts), request=body.request, source=body.source
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    # the adaptive model learns from every request (after the response is sent)
    background.add_task(service.safe_catch_up, body.patient_id)
    return req


@app.get("/requests", response_model=list[schemas.RequestOut])
def list_requests(
    patient_id: int | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    limit: int = Query(10000, ge=1, le=100000),
    db: Session = Depends(get_db),
):
    stmt = select(models.Request)
    if patient_id is not None:
        stmt = stmt.where(models.Request.patient_id == patient_id)
    if from_ is not None:
        stmt = stmt.where(models.Request.ts >= _wall(from_))
    if to is not None:
        stmt = stmt.where(models.Request.ts <= _wall(to))
    return db.scalars(stmt.order_by(models.Request.ts).limit(limit)).all()


# ----------------------------------------------------------------- predict
@app.post("/predict", response_model=schemas.PredictOut)
def run_predict(body: schemas.PredictIn, db: Session = Depends(get_db)):
    _patient_or_404(db, body.patient_id)
    try:
        return predict(body.patient_id, _wall(body.ts))
    except FileNotFoundError as e:  # model not trained yet
        raise HTTPException(503, str(e))


# ------------------------------------------------------------------ export
@app.get("/export/requests.csv")
def export_requests_csv(patient_id: int | None = None, db: Session = Depends(get_db)):
    stmt = select(models.Request).order_by(models.Request.ts)
    if patient_id is not None:
        stmt = stmt.where(models.Request.patient_id == patient_id)

    def rows():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "patient_id", "ts", "request", "source"])
        yield buf.getvalue()
        for r in db.scalars(stmt).yield_per(500):
            buf.seek(0)
            buf.truncate(0)
            writer.writerow([r.id, r.patient_id, r.ts.isoformat(), r.request, r.source])
            yield buf.getvalue()

    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="requests.csv"'},
    )


# =================================================================================== added
# Everything below is new. The endpoints above keep their exact contract.

@app.post("/patients", response_model=schemas.PatientOut, status_code=201)
def create_patient(body: schemas.PatientIn, db: Session = Depends(get_db)):
    p = models.Patient(name=body.name, age=body.age, notes=body.notes)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ------------------------------------------------------------------- B1 windows
def _window_row(w: schemas.WindowIn) -> dict:
    msg = w.message
    pid, start_ms = None, None
    if w.key and "|" in w.key:
        a, b = w.key.split("|", 1)
        pid = schemas.coerce_patient_id(a)
        try:
            start_ms = int(float(b))
        except ValueError:
            start_ms = None
    if pid is None or not str(pid).isdigit():
        pid = schemas.coerce_patient_id(msg.User_ID)
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        raise HTTPException(422, f"cannot tell the patient from key {w.key!r} / User_ID {msg.User_ID!r}")
    if msg.Max_Gaze_Deviation + 1e-9 < msg.Mean_Gaze_Deviation:
        raise HTTPException(422, "'Max_Gaze_Deviation' cannot be smaller than 'Mean_Gaze_Deviation'")
    if msg.Max_Fixation_Duration + 1e-9 < msg.Avg_Fixation_Duration:
        raise HTTPException(422, "'Max_Fixation_Duration' cannot be smaller than 'Avg_Fixation_Duration'")
    if start_ms is None:  # no key: the window just ended
        now = datetime.now(timezone.utc).timestamp()
        start_ms = int((now // 10) * 10 - 10) * 1000
    row = {
        "patient_id": pid, "start": datetime.fromtimestamp(start_ms / 1000, timezone.utc).replace(tzinfo=None),
        "user_id": msg.User_ID, "time_window": msg.Time_Window, "simulated": w.simulated,
        "valid": w.valid, "session_id": w.session_id,
    }
    for f, c in models.WINDOW_COLUMNS.items():
        row[c] = getattr(msg, f)
    return row


@app.post("/windows", status_code=201)
def post_windows(body: schemas.WindowIn | list[schemas.WindowIn], db: Session = Depends(get_db)):
    """Store 10 s webcam windows. Same envelope GazePad sends to the B1 server
    ({message, key, simulated}), optionally with valid and session_id. Re-sent windows are ignored."""
    items = body if isinstance(body, list) else [body]
    rows = [_window_row(w) for w in items]
    if not rows:
        return {"inserted": 0, "duplicates": 0}
    _ensure_patients(db, {r["patient_id"] for r in rows})
    inserted = dup = 0
    for r in rows:
        exists = db.scalar(select(models.GazeWindow.id).where(
            models.GazeWindow.patient_id == r["patient_id"], models.GazeWindow.start == r["start"]))
        if exists:
            dup += 1
            continue
        db.add(models.GazeWindow(**r))
        inserted += 1
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "duplicate window")
    return {"inserted": inserted, "duplicates": dup}


def _window_out(w: models.GazeWindow) -> dict:
    return {
        "patient_id": w.patient_id, "start": iso(w.start.replace(tzinfo=timezone.utc).timestamp()),
        "session_id": w.session_id, "simulated": w.simulated, "valid": w.valid, "message": window_message(w),
    }


@app.get("/windows")
def list_windows(
    patient_id: str,
    minutes: int = Query(60, ge=1, le=60 * 24 * 14),
    limit: int = Query(2000, ge=1, le=20000),
    db: Session = Depends(get_db),
):
    pid = _pid(patient_id)
    since = _now_utc() - timedelta(minutes=minutes)
    rows = db.scalars(
        select(models.GazeWindow)
        .where(models.GazeWindow.patient_id == pid, models.GazeWindow.start >= since)
        .order_by(models.GazeWindow.start.desc()).limit(limit)
    ).all()
    return [_window_out(w) for w in reversed(rows)]


@app.get("/windows/latest")
def latest_window(patient_id: str, db: Session = Depends(get_db)):
    pid = _pid(patient_id)
    w = db.scalars(select(models.GazeWindow).where(models.GazeWindow.patient_id == pid)
                   .order_by(models.GazeWindow.start.desc()).limit(1)).first()
    if w is None:
        raise HTTPException(404, "no window for that patient yet")
    return _window_out(w)


# -------------------------------------------------------------------- sessions
def _session_out(s: models.CaptureSession, n_events: int | None = None) -> dict:
    t = lambda d: iso(d.replace(tzinfo=timezone.utc).timestamp()) if d else None
    out = {"id": s.id, "patient_id": s.patient_id, "app": s.app, "mode": s.mode, "started_at": t(s.started_at),
           "last_seen_at": t(s.last_seen_at), "ended_at": t(s.ended_at), "params": s.params or {}, "client": s.client or {}}
    if n_events is not None:
        out["events"] = n_events
    return out


@app.post("/sessions", status_code=201)
def upsert_session(body: schemas.SessionIn, db: Session = Depends(get_db)):
    """Create (or refresh) a GazePad session with every selected setting. Idempotent on id."""
    _ensure_patients(db, {body.patient_id})
    s = db.get(models.CaptureSession, body.id)
    now = _now_utc()
    if s is None:
        s = models.CaptureSession(id=body.id, patient_id=body.patient_id, app=body.app, mode=body.mode,
                                  started_at=_wall(body.started_at) if body.started_at else now,
                                  last_seen_at=now, params=body.params, client=body.client)
        db.add(s)
    else:
        s.patient_id, s.app, s.mode, s.last_seen_at = body.patient_id, body.app, body.mode or s.mode, now
        s.params = {**(s.params or {}), **body.params}
        s.client = {**(s.client or {}), **body.client}
    db.commit()
    return _session_out(s)


@app.patch("/sessions/{session_id}")
def patch_session(session_id: str, body: schemas.SessionPatch, db: Session = Depends(get_db)):
    s = db.get(models.CaptureSession, session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    if body.mode:
        s.mode = body.mode
    if body.params:
        s.params = {**(s.params or {}), **body.params}
    if body.client:
        s.client = {**(s.client or {}), **body.client}
    if body.ended_at:
        s.ended_at = _wall(body.ended_at)
    s.last_seen_at = _now_utc()
    db.commit()
    return _session_out(s)


@app.post("/sessions/{session_id}/events", status_code=201)
def post_session_events(session_id: str, body: schemas.SessionEventIn | list[schemas.SessionEventIn],
                        db: Session = Depends(get_db)):
    s = db.get(models.CaptureSession, session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    items = body if isinstance(body, list) else [body]
    now = _now_utc()
    for e in items:
        db.add(models.SessionEvent(session_id=s.id, patient_id=s.patient_id, ts=_wall(e.ts) if e.ts else now,
                                   type=e.type, data=e.data))
        if e.type == "setting_changed" and isinstance(e.data, dict) and "name" in e.data:
            s.params = {**(s.params or {}), e.data["name"]: e.data.get("value")}
        if e.type == "mode_changed" and isinstance(e.data, dict) and e.data.get("mode"):
            s.mode = e.data["mode"]
        if e.type == "ended":
            s.ended_at = _wall(e.ts) if e.ts else now
    s.last_seen_at = now
    db.commit()
    return {"inserted": len(items)}


@app.get("/sessions")
def list_sessions(patient_id: str | None = None, limit: int = Query(50, ge=1, le=1000), db: Session = Depends(get_db)):
    stmt = select(models.CaptureSession)
    if patient_id is not None:
        stmt = stmt.where(models.CaptureSession.patient_id == _pid(patient_id))
    rows = db.scalars(stmt.order_by(models.CaptureSession.started_at.desc()).limit(limit)).all()
    counts = dict(db.execute(select(models.SessionEvent.session_id, func.count())
                             .where(models.SessionEvent.session_id.in_([r.id for r in rows]))
                             .group_by(models.SessionEvent.session_id)).all()) if rows else {}
    return [_session_out(r, counts.get(r.id, 0)) for r in rows]


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    s = db.get(models.CaptureSession, session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    events = db.scalars(select(models.SessionEvent).where(models.SessionEvent.session_id == session_id)
                        .order_by(models.SessionEvent.ts)).all()
    out = _session_out(s, len(events))
    out["event_log"] = [{"ts": iso(e.ts.replace(tzinfo=timezone.utc).timestamp()), "type": e.type, "data": e.data}
                        for e in events]
    out["windows"] = db.scalar(select(func.count()).select_from(models.GazeWindow)
                               .where(models.GazeWindow.session_id == session_id))
    return out


# -------------------------------------------------------------- adaptive insights
def _insights(db: Session, pid: int, ts: datetime | None):
    _patient_or_404(db, pid)
    t = _wall(ts).replace(tzinfo=timezone.utc).timestamp() if ts else None
    out = service.insights(db, pid, t)
    p = db.get(models.Patient, pid)
    out["patient"] = {"id": p.id, "name": p.name}
    return out


@app.get("/insights")
def get_insights(patient_id: str, ts: datetime | None = None, db: Session = Depends(get_db)):
    """What this patient is likely to need now, why, their webcam state, and care actions.
    `suggestions`, `confused` and `confusion_score` have the same shape as POST /predict."""
    return _insights(db, _pid(patient_id), ts)


@app.post("/insights")
def post_insights(body: schemas.InsightsIn, db: Session = Depends(get_db)):
    return _insights(db, body.patient_id, body.ts)


@app.post("/feedback", status_code=201)
def post_feedback(body: schemas.FeedbackIn, db: Session = Depends(get_db)):
    """Caregiver marked a need as done (met) or dismissed it (not needed)."""
    _patient_or_404(db, body.patient_id)
    fb = models.CareFeedback(patient_id=body.patient_id, ts=_wall(body.ts) if body.ts else _now_utc(),
                             label=body.label, action=body.action, suggestion_id=body.suggestion_id,
                             caregiver=body.caregiver, note=body.note)
    db.add(fb)
    db.commit()
    return {"id": fb.id, "patient_id": fb.patient_id, "ts": iso(fb.ts.replace(tzinfo=timezone.utc).timestamp()),
            "label": fb.label, "action": fb.action}


@app.get("/feedback")
def list_feedback(patient_id: str, hours: int = Query(24, ge=1, le=24 * 60), db: Session = Depends(get_db)):
    pid = _pid(patient_id)
    rows = db.scalars(select(models.CareFeedback).where(
        models.CareFeedback.patient_id == pid, models.CareFeedback.ts >= _now_utc() - timedelta(hours=hours))
        .order_by(models.CareFeedback.ts)).all()
    return [{"id": r.id, "ts": iso(r.ts.replace(tzinfo=timezone.utc).timestamp()), "label": r.label,
             "action": r.action, "suggestion_id": r.suggestion_id, "caregiver": r.caregiver, "note": r.note} for r in rows]


@app.get("/model/stats")
def model_stats(patient_id: str, db: Session = Depends(get_db)):
    """How well the adaptive model has done for this patient, over time (all out-of-sample)."""
    pid = _pid(patient_id)
    _patient_or_404(db, pid)
    service.catch_up(db, pid)
    ev = db.scalars(select(models.PredictionEval).where(models.PredictionEval.patient_id == pid)
                    .order_by(models.PredictionEval.ts)).all()
    n = len(ev)

    def acc(rows):
        k = len(rows)
        return {"n": k, "top1": round(sum(r.top1 for r in rows) / k, 4) if k else None,
                "top3": round(sum(r.top3 for r in rows) / k, 4) if k else None}

    by_day = {}
    for r in ev:
        by_day.setdefault(r.ts.date().isoformat(), []).append(r)
    curve, run = [], []
    for i, r in enumerate(ev):  # rolling accuracy over the last 20 requests
        run.append(r)
        if len(run) > 20:
            run.pop(0)
        if i % max(1, n // 200) == 0 or i == n - 1:
            curve.append({"i": i + 1, "ts": iso(r.ts.replace(tzinfo=timezone.utc).timestamp()), **acc(run)})
    labels = ["water", "food", "help", "pain", "bathroom", "rest"]
    confusion = {a: {p: 0 for p in labels} for a in labels}
    for r in ev:
        confusion[r.actual][r.predicted] += 1
    expert_acc = {}
    for r in ev:
        for k, v in (r.experts or {}).items():
            e = expert_acc.setdefault(k, [0, 0.0])
            e[0] += 1
            e[1] += v
    state = db.get(models.ModelState, f"patient:{pid}")
    return {
        "patient_id": pid, "n": n, "overall": acc(ev), "last_20": acc(ev[-20:]), "last_50": acc(ev[-50:]),
        "webcam": acc([r for r in ev if r.webcam_available]), "no_webcam": acc([r for r in ev if not r.webcam_available]),
        "by_day": [{"day": d, **acc(rs)} for d, rs in sorted(by_day.items())][-30:],
        "rolling": curve, "confusion": confusion,
        "expert_mean_p_actual": {k: round(v[1] / v[0], 4) for k, v in expert_acc.items()},
        "weights": (state.data or {}).get("weights") if state else None,
        "updated_at": iso(state.updated_at.replace(tzinfo=timezone.utc).timestamp()) if state else None,
    }


@app.get("/model/evals")
def model_evals(patient_id: str, limit: int = Query(100, ge=1, le=5000), db: Session = Depends(get_db)):
    pid = _pid(patient_id)
    rows = db.scalars(select(models.PredictionEval).where(models.PredictionEval.patient_id == pid)
                      .order_by(models.PredictionEval.ts.desc()).limit(limit)).all()
    return [{"ts": iso(r.ts.replace(tzinfo=timezone.utc).timestamp()), "actual": r.actual, "predicted": r.predicted,
             "p_actual": r.p_actual, "rank": r.rank, "top1": r.top1, "top3": r.top3, "experts": r.experts,
             "webcam_available": r.webcam_available, "source": r.source} for r in rows]


@app.post("/model/retrain")
def model_retrain(patient_id: str | None = None, db: Session = Depends(get_db)):
    """Forget the adaptive state and replay history in time order (one patient, or everyone)."""
    if patient_id is None:
        return {"learned": service.retrain_all()}
    pid = _pid(patient_id)
    _patient_or_404(db, pid)
    return {"learned": {pid: service.retrain(db, pid)}}


# --------------------------------------------------------------------- exports
def _csv(name: str, header: list[str], rows_iter):
    def gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        yield buf.getvalue()
        for row in rows_iter:
            buf.seek(0)
            buf.truncate(0)
            w.writerow(row)
            yield buf.getvalue()
    return StreamingResponse(gen(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/export/windows.csv")
def export_windows_csv(patient_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(models.GazeWindow).order_by(models.GazeWindow.patient_id, models.GazeWindow.start)
    if patient_id is not None:
        stmt = stmt.where(models.GazeWindow.patient_id == _pid(patient_id))
    fields = list(models.WINDOW_COLUMNS)
    return _csv("windows.csv", ["patient_id", "start", "session_id", "simulated", "valid", "User_ID", "Time_Window", *fields],
                ([w.patient_id, w.start.isoformat(), w.session_id, w.simulated, w.valid, w.user_id, w.time_window,
                  *[getattr(w, models.WINDOW_COLUMNS[f]) for f in fields]] for w in db.scalars(stmt).yield_per(1000)))


@app.get("/export/sessions.csv")
def export_sessions_csv(db: Session = Depends(get_db)):
    rows = db.scalars(select(models.CaptureSession).order_by(models.CaptureSession.started_at)).all()
    keys = sorted({k for r in rows for k in (r.params or {})})
    return _csv("sessions.csv", ["id", "patient_id", "app", "mode", "started_at", "ended_at", *keys],
                ([r.id, r.patient_id, r.app, r.mode, r.started_at.isoformat(), r.ended_at.isoformat() if r.ended_at else "",
                  *[(r.params or {}).get(k, "") for k in keys]] for r in rows))


@app.get("/export/evals.csv")
def export_evals_csv(patient_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(models.PredictionEval).order_by(models.PredictionEval.patient_id, models.PredictionEval.ts)
    if patient_id is not None:
        stmt = stmt.where(models.PredictionEval.patient_id == _pid(patient_id))
    return _csv("evals.csv", ["patient_id", "ts", "actual", "predicted", "p_actual", "rank", "top1", "top3",
                              "webcam_available", "source"],
                ([r.patient_id, r.ts.isoformat(), r.actual, r.predicted, r.p_actual, r.rank, r.top1, r.top3,
                  r.webcam_available, r.source] for r in db.scalars(stmt).yield_per(1000)))


@app.get("/export/feedback.csv")
def export_feedback_csv(db: Session = Depends(get_db)):
    rows = db.scalars(select(models.CareFeedback).order_by(models.CareFeedback.ts)).all()
    return _csv("feedback.csv", ["id", "patient_id", "ts", "label", "action", "suggestion_id", "caregiver", "note"],
                ([r.id, r.patient_id, r.ts.isoformat(), r.label, r.action, r.suggestion_id, r.caregiver, r.note] for r in rows))
