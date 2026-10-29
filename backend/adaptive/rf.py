"""The existing RandomForest (model/artifacts/model.joblib) as one expert in the ensemble.

It was trained offline on simulated routines; the ensemble decides per patient how much to
trust it, so it helps where it is right and fades where real behaviour differs.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .config import N, local_dt
from .data import NONE_ZONE, PatientData


class StaticRF:
    def __init__(self, path=None):
        self.path = path
        self._bundle = None
        self._failed = False

    def _get(self):
        if self._bundle is None and not self._failed:
            try:
                import joblib
                from model.predict import MODEL_PATH
                self._bundle = joblib.load(self.path or MODEL_PATH)
                for m in [*self._bundle["models"].values(), self._bundle["global_model"]]:
                    m.n_jobs = 1  # one row at a time: thread start-up would dominate
            except Exception:
                self._failed = True
        return self._bundle

    @property
    def available(self) -> bool:
        return self._get() is not None

    def __call__(self, h: PatientData, t: float, ctx: dict, patient_id=None):
        b = self._get()
        if b is None or h.n_requests < 1:
            return None, []
        from model.features import build_features

        # the RF reads hour/minute/weekday from the timestamp: give it the patient's wall clock
        off = local_dt(t).utcoffset() or timedelta(0)
        shift = np.timedelta64(int(off.total_seconds()), "s")
        to64 = lambda a: (np.asarray(a) * 1e9).astype("int64").astype("datetime64[ns]") + shift
        ts = (np.datetime64(int(t * 1e9), "ns") + shift).astype("datetime64[ms]").astype(object)
        g_zone = np.where(h.gaze_z == NONE_ZONE, N, h.gaze_z)
        x, _ = build_features(ts, to64(h.req_t), h.req_y, to64(h.gaze_t), g_zone, h.gaze_d)
        model = b["models"].get(patient_id, b["global_model"]) if patient_id is not None else b["global_model"]
        out = np.zeros(N)
        out[model.classes_.astype(int)] = _fast_proba(model, x)
        return out, []


def _fast_proba(forest, x: np.ndarray) -> np.ndarray:
    """predict_proba for one row without joblib's per-tree overhead (about 30x faster)."""
    try:
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        acc = np.zeros(len(forest.classes_))
        for est in forest.estimators_:
            v = est.tree_.value[est.tree_.apply(X)[0], 0]
            acc += v / v.sum()
        return acc / len(forest.estimators_)
    except Exception:
        return forest.predict_proba(np.asarray(x).reshape(1, -1))[0]
