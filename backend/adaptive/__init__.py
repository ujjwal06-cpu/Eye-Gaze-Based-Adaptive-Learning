"""Adaptive, per-patient need prediction for the eye-gaze request system.

Pure Python + numpy: no database, no web framework. The backend (app/adaptive_service.py)
and the ward simulator (sim/ward.py) both feed it a PatientData history.

    from adaptive import Engine, PatientData
    engine = Engine()
    state = engine.new_state()
    insight = engine.predict(data, state, t)        # what the patient may need now, and why
    record = engine.learn(data, state, t, "water")  # prequential update when a request arrives

See adaptive/README.md for the design.
"""
from .config import LABELS, LABEL_IDX, WIN_FIELDS
from .data import PatientData
from .engine import Engine, ENGINE_VERSION

__all__ = ["Engine", "ENGINE_VERSION", "PatientData", "LABELS", "LABEL_IDX", "WIN_FIELDS"]
