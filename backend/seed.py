"""Create tables and insert 3 demo patients. Safe to re-run.

    python seed.py              # 3 patients (ids 1-3 on a fresh DB)
    python seed.py --load-sim   # also load model/data/{requests,gaze}.csv (from model.simulate)

The simulated data uses patient ids 1-3, so --load-sim expects a fresh database.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import func, insert, select

from app import models
from app.database import Base, SessionLocal, engine

PATIENTS = [
    {"name": "Asha Rao", "age": 67, "notes": "Post-stroke, limited speech"},
    {"name": "Ravi Kumar", "age": 54, "notes": "ALS, uses gaze input"},
    {"name": "Meena Iyer", "age": 72, "notes": "Locked-in syndrome"},
]
SIM_DIR = Path(__file__).parent / "model" / "data"
CHUNK = 10_000


def _insert_chunks(db, table, rows):
    for i in range(0, len(rows), CHUNK):
        db.execute(insert(table), rows[i : i + CHUNK])


def load_sim(db) -> None:
    if db.scalar(select(func.count()).select_from(models.Request)) or db.scalar(
        select(func.count()).select_from(models.GazeEvent)
    ):
        print("Requests/gaze already present; skipping --load-sim.")
        return
    req = pd.read_csv(SIM_DIR / "requests.csv")
    gaze = pd.read_csv(SIM_DIR / "gaze.csv")
    req_rows = [
        {"patient_id": int(r.patient_id), "ts": datetime.fromisoformat(r.ts),
         "request": r.request, "source": r.source}
        for r in req.itertuples(index=False)
    ]
    gaze_rows = [
        {"patient_id": int(r.patient_id), "ts": datetime.fromisoformat(r.ts), "x": float(r.x),
         "y": float(r.y), "zone": r.zone if isinstance(r.zone, str) else None,
         "dwell_ms": float(r.dwell_ms), "confidence": float(r.confidence)}
        for r in gaze.itertuples(index=False)
    ]
    _insert_chunks(db, models.Request, req_rows)
    _insert_chunks(db, models.GazeEvent, gaze_rows)
    db.commit()
    print(f"Loaded {len(req_rows)} requests and {len(gaze_rows)} gaze events.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-sim", action="store_true", help="load simulated requests/gaze CSVs")
    args = ap.parse_args()

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        existing = set(db.scalars(select(models.Patient.name)).all())
        added = 0
        for p in PATIENTS:
            if p["name"] not in existing:
                db.add(models.Patient(**p))
                added += 1
        db.commit()
        print(f"Seeded {added} patient(s).")
        if args.load_sim:
            load_sim(db)


if __name__ == "__main__":
    main()
