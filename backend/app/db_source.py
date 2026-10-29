"""History source for model.predict, backed by this app's database."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
from sqlalchemy import select

from model.features import LABEL_IDX, zone_codes

from . import models
from .database import SessionLocal

REQUEST_LOOKBACK = timedelta(days=7)


class DBSource:
    """Implements the data-source interface documented in model/predict.py."""

    def requests(self, patient_id, until):
        until = until.to_pydatetime() if hasattr(until, "to_pydatetime") else until
        with SessionLocal() as db:
            rows = db.execute(
                select(models.Request.ts, models.Request.request)
                .where(
                    models.Request.patient_id == patient_id,
                    models.Request.ts < until,
                    models.Request.ts >= until - REQUEST_LOOKBACK,
                )
                .order_by(models.Request.ts)
            ).all()
        ts = np.array([r[0] for r in rows], dtype="datetime64[ns]")
        codes = np.array([LABEL_IDX[r[1]] for r in rows], dtype=np.int64)
        return ts, codes

    def gaze(self, patient_id, since, until):
        since = since.to_pydatetime() if hasattr(since, "to_pydatetime") else since
        until = until.to_pydatetime() if hasattr(until, "to_pydatetime") else until
        with SessionLocal() as db:
            rows = db.execute(
                select(models.GazeEvent.ts, models.GazeEvent.zone, models.GazeEvent.dwell_ms)
                .where(
                    models.GazeEvent.patient_id == patient_id,
                    models.GazeEvent.ts > since,
                    models.GazeEvent.ts <= until,
                )
                .order_by(models.GazeEvent.ts)
            ).all()
        ts = np.array([r[0] for r in rows], dtype="datetime64[ns]")
        zones = zone_codes([r[1] for r in rows])
        dwell = np.array([r[2] for r in rows], dtype=float)
        return ts, zones, dwell
