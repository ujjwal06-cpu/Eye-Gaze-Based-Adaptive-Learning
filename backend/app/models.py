from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    gaze_events: Mapped[list["GazeEvent"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    requests: Mapped[list["Request"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class GazeEvent(Base):
    """Contract: {patient_id, ts, x, y, zone, dwell_ms, confidence}."""

    __tablename__ = "gaze_events"
    __table_args__ = (Index("ix_gaze_events_patient_ts", "patient_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # patient wall-clock, naive
    x: Mapped[float] = mapped_column(Float, nullable=False)          # 0-1
    y: Mapped[float] = mapped_column(Float, nullable=False)          # 0-1
    zone: Mapped[str | None] = mapped_column(String(20), nullable=True)  # a label, or null
    dwell_ms: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="gaze_events")


class Request(Base):
    """Contract: {patient_id, ts, request (label), source ("gaze" | "simulated")}."""

    __tablename__ = "requests"
    __table_args__ = (Index("ix_requests_patient_ts", "patient_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    request: Mapped[str] = mapped_column(String(20), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="gaze")

    patient: Mapped[Patient] = relationship(back_populates="requests")


# ============================================================================================
# Added for adaptive prediction. New tables only: the three tables above are unchanged, so an
# existing app.db keeps working (create_all adds the new tables on startup).
# ============================================================================================
from sqlalchemy import JSON, Boolean, UniqueConstraint  # noqa: E402

# B1 field name (the contract, unchanged) -> column name
WINDOW_COLUMNS = {
    "Mean_Gaze_Deviation": "mean_gaze_deviation",
    "Max_Gaze_Deviation": "max_gaze_deviation",
    "Blink_Rate": "blink_rate",
    "Avg_Fixation_Duration": "avg_fixation_duration",
    "Max_Fixation_Duration": "max_fixation_duration",
    "Head_Movement": "head_movement",
    "Gaze_Stability": "gaze_stability",
    "Emergency_Button": "emergency_button",
}


class GazeWindow(Base):
    """One 10 s webcam window: exactly the B1 message GazePad sends to the B1 server, plus metadata."""

    __tablename__ = "gaze_windows"
    __table_args__ = (
        UniqueConstraint("patient_id", "start", name="uq_gaze_windows_patient_start"),
        Index("ix_gaze_windows_patient_start", "patient_id", "start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    start: Mapped[datetime] = mapped_column(DateTime, nullable=False)       # window start, UTC
    user_id: Mapped[str] = mapped_column(String(32), nullable=False)         # B1 User_ID
    time_window: Mapped[str] = mapped_column(String(20), nullable=False)     # B1 Time_Window
    mean_gaze_deviation: Mapped[float] = mapped_column(Float, nullable=False)
    max_gaze_deviation: Mapped[float] = mapped_column(Float, nullable=False)
    blink_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_fixation_duration: Mapped[float] = mapped_column(Float, nullable=False)
    max_fixation_duration: Mapped[float] = mapped_column(Float, nullable=False)
    head_movement: Mapped[float] = mapped_column(Float, nullable=False)
    gaze_stability: Mapped[float] = mapped_column(Float, nullable=False)
    emergency_button: Mapped[int] = mapped_column(Integer, nullable=False)
    simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    valid: Mapped[int | None] = mapped_column(Integer, nullable=True)        # gaze slots with data, of 100
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CaptureSession(Base):
    """One GazePad run: every setting that was selected (mode, dwell, voice, B1, screen size ...)."""

    __tablename__ = "capture_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)            # client-generated UUID
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    app: Mapped[str] = mapped_column(String(32), nullable=False, default="gazepad")
    mode: Mapped[str | None] = mapped_column(String(20), nullable=True)      # webcam | mock
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)                 # selected settings
    client: Mapped[dict] = mapped_column(JSON, default=dict)                 # screen, pad size, browser


class SessionEvent(Base):
    """Things that happened in a session: calibration, setting changes, face lost, mode changes."""

    __tablename__ = "session_events"
    __table_args__ = (Index("ix_session_events_session_ts", "session_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("capture_sessions.id"), nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    type: Mapped[str] = mapped_column(String(40), nullable=False)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class CareFeedback(Base):
    """Caregiver response to a suggestion: done (need met) or dismiss (not needed)."""

    __tablename__ = "care_feedback"
    __table_args__ = (Index("ix_care_feedback_patient_ts", "patient_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    suggestion_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    caregiver: Mapped[str | None] = mapped_column(String(80), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class PredictionEval(Base):
    """For every learned request: what the model predicted 15 s earlier, and whether it was right."""

    __tablename__ = "prediction_evals"
    __table_args__ = (Index("ix_prediction_evals_patient_ts", "patient_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), nullable=False)
    request_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    actual: Mapped[str] = mapped_column(String(20), nullable=False)
    predicted: Mapped[str] = mapped_column(String(20), nullable=False)
    p_actual: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    top1: Mapped[bool] = mapped_column(Boolean, nullable=False)
    top3: Mapped[bool] = mapped_column(Boolean, nullable=False)
    experts: Mapped[dict] = mapped_column(JSON, default=dict)                # each expert's P(actual)
    webcam_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    engine_version: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelState(Base):
    """Adaptive model state as JSON: key "patient:<id>" or "global"."""

    __tablename__ = "model_state"

    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
