import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

Label = Literal["water", "food", "help", "pain", "bathroom", "rest"]
Source = Literal["gaze", "simulated"]


def coerce_patient_id(v: Any) -> Any:
    """GazePad's default patient id is "p1" and B1 uses "USER001": all of them mean patient 1."""
    if isinstance(v, str) and not v.strip().isdigit():
        m = re.search(r"\d+", v)
        if m:
            return int(m.group())
    return v


PatientRef = Annotated[int, BeforeValidator(coerce_patient_id)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- patients ----
class PatientOut(ORMModel):
    id: int
    name: str
    age: int | None
    notes: str | None


# ---- gaze ----
class GazeEventIn(BaseModel):
    patient_id: PatientRef
    ts: datetime
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    zone: Label | None = None
    dwell_ms: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class GazeBatchIn(BaseModel):
    events: list[GazeEventIn]


class GazeEventOut(ORMModel):
    id: int
    patient_id: int
    ts: datetime
    x: float
    y: float
    zone: Label | None
    dwell_ms: float
    confidence: float


# ---- requests ----
class RequestIn(BaseModel):
    patient_id: PatientRef
    ts: datetime
    request: Label
    source: Source = "gaze"


class RequestOut(ORMModel):
    id: int
    patient_id: int
    ts: datetime
    request: Label
    source: Source


# ---- predict ----
class PredictIn(BaseModel):
    patient_id: PatientRef
    ts: datetime


class Suggestion(BaseModel):
    label: Label
    prob: float


class PredictOut(BaseModel):
    suggestions: list[Suggestion]
    confused: bool
    confusion_score: float


# =========================================================================== added
class PatientIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    age: int | None = Field(None, ge=0, le=130)
    notes: str | None = None


# ---- B1 windows (same envelope GazePad posts to the B1 server) ----
class B1Message(BaseModel):
    """Exactly the 10 B1 fields. Field names are the B1 contract and must not change."""
    model_config = ConfigDict(extra="forbid")

    User_ID: str = Field(min_length=1)
    Time_Window: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}$")
    Mean_Gaze_Deviation: float = Field(ge=0)
    Max_Gaze_Deviation: float = Field(ge=0)
    Blink_Rate: int = Field(ge=0, strict=False)
    Avg_Fixation_Duration: float = Field(ge=0)
    Max_Fixation_Duration: float = Field(ge=0, le=10)
    Head_Movement: float = Field(ge=0)
    Gaze_Stability: float = Field(ge=0, le=1)
    Emergency_Button: int = Field(ge=0)


class WindowIn(BaseModel):
    message: B1Message
    key: str | None = None            # "patientId|windowStartEpochMs", as sent to the B1 server
    simulated: bool = False
    valid: int | None = Field(None, ge=0, le=100)
    session_id: str | None = None


class WindowOut(BaseModel):
    patient_id: int
    start: str
    session_id: str | None
    simulated: bool
    valid: int | None
    message: dict


# ---- sessions: every parameter selected in GazePad ----
class SessionIn(BaseModel):
    id: str = Field(min_length=8, max_length=64)
    patient_id: PatientRef
    app: str = "gazepad"
    mode: str | None = None
    started_at: datetime | None = None
    params: dict = Field(default_factory=dict)
    client: dict = Field(default_factory=dict)


class SessionPatch(BaseModel):
    mode: str | None = None
    ended_at: datetime | None = None
    params: dict | None = None        # merged into the stored params
    client: dict | None = None


class SessionEventIn(BaseModel):
    ts: datetime | None = None
    type: str = Field(min_length=1, max_length=40)
    data: dict = Field(default_factory=dict)


# ---- caregiver feedback ----
class FeedbackIn(BaseModel):
    patient_id: PatientRef
    label: Label
    action: Literal["done", "dismiss"]
    ts: datetime | None = None
    suggestion_id: str | None = None
    caregiver: str | None = None
    note: str | None = None


class InsightsIn(BaseModel):
    patient_id: PatientRef
    ts: datetime | None = None
