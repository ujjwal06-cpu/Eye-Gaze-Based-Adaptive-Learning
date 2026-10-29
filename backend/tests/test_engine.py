"""Adaptive engine unit tests (no database)."""
import numpy as np

from adaptive import Engine, PatientData
from adaptive.config import LEAD_S


def _routine_data(days=8):
    t0 = 1_790_000_000.0
    reqs = []
    for d in range(days):
        b = t0 + d * 86400
        reqs += [(b, "water"), (b + 3 * 3600, "food"), (b + 6 * 3600, "rest")]
    return t0, reqs


def test_view_never_leaks_the_future():
    t0, reqs = _routine_data()
    data = PatientData.from_records(requests=reqs, windows=[(t0 + 100, [1] * 8)])
    h = data.view(t0 + 3 * 3600)
    assert h.n_requests == 1 and len(h.win_t) == 1
    assert data.view(t0 + 105).win_t.size == 0  # a window only counts once it has ended


def test_learn_scores_before_updating():
    t0, reqs = _routine_data()
    data = PatientData.from_records(requests=reqs)
    e = Engine()
    s, g = e.new_state(), e.new_global_state()
    recs = [e.learn(data, s, t, l, g) for t, l in reqs]
    assert recs[0]["rank"] == 6  # nothing known yet: a uniform guess, ties count against the model
    assert all(r["top1"] for r in recs[-6:])  # the routine has been learned
    assert s["n"] == len(reqs) and s["last_t"] == reqs[-1][0]


def test_routine_change_is_relearned():
    t0, reqs = _routine_data(10)
    shifted = [(t + 10 * 86400 + 2 * 3600, l) for t, l in reqs]  # same needs, 2 h later from day 10
    data = PatientData.from_records(requests=reqs + shifted)
    e = Engine()
    s, g = e.new_state(), e.new_global_state()
    hits = [e.learn(data, s, t, l, g)["top1"] for t, l in reqs + shifted]
    assert np.mean(hits[-9:]) > np.mean(hits[30:33])


def test_webcam_distress_moves_prediction_to_pain():
    t0, reqs = _routine_data(6)
    calm = [(t0 + 6 * 86400 - 3600 + 10 * i, [4, 8, 3, 0.3, 0.8, 1.0 + 0.1 * (i % 4), 0.35, 0]) for i in range(300)]
    q = t0 + 6 * 86400 + 3600  # an hour after the usual water: nothing due from the routine
    agitated = [(q - 90 + 10 * i, [6, 11, 6, 0.2, 0.5, 15.0, 0.18, 0]) for i in range(8)]
    data = PatientData.from_records(requests=reqs, windows=calm + agitated)
    out = Engine().predict(data, Engine().new_state(), q)
    assert out["webcam"]["available"] and out["webcam"]["states"]["distress"] > 0.8
    top2 = [n["label"] for n in out["needs"][:2]]
    assert "pain" in top2 or "help" in top2
    assert out["alert"]["level"] == "urgent"


def test_simulated_ward_learns_and_webcam_helps():
    from sim.ward import evaluate

    rep = evaluate(days=8, seed=3, quiet=True, write=False)
    o = rep["overall"]
    assert o["adaptive_top3"] >= o["frequency_top3"]
    assert rep["by_cause"]["episode"]["adaptive_top1"] > rep["by_cause"]["episode"]["no_webcam_top1"]
    assert LEAD_S > 0
