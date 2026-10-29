"""API tests: the original contract is unchanged, and the new storage / adaptive endpoints work.

    cd backend && python -m pytest -q
"""
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["ADAPTIVE_CATCH_UP_ON_START"] = "0"
os.environ["ADAPTIVE_STATIC_RF"] = "0"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

B1_FIELDS = ["User_ID", "Time_Window", "Mean_Gaze_Deviation", "Max_Gaze_Deviation", "Blink_Rate",
             "Avg_Fixation_Duration", "Max_Fixation_Duration", "Head_Movement", "Gaze_Stability", "Emergency_Button"]


def z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def window(pid, start: datetime, **over):
    msg = {"User_ID": f"USER{pid:03d}", "Time_Window": f"{start:%H:%M:%S}-{start + timedelta(seconds=10):%H:%M:%S}",
           "Mean_Gaze_Deviation": 4.1, "Max_Gaze_Deviation": 8.7, "Blink_Rate": 3, "Avg_Fixation_Duration": 0.34,
           "Max_Fixation_Duration": 0.8, "Head_Movement": 1.2, "Gaze_Stability": 0.42, "Emergency_Button": 0, **over}
    return {"message": msg, "key": f"p{pid}|{int(start.timestamp() * 1000)}", "simulated": False}


@pytest.fixture(scope="module")
def c():
    with TestClient(app) as client:
        for name in ("A", "B"):
            client.post("/patients", json={"name": name})
        yield client


def test_original_contract(c):
    now = datetime.now(timezone.utc)
    r = c.post("/requests", json={"patient_id": 1, "ts": z(now), "request": "water", "source": "gaze"})
    assert r.status_code == 201
    assert set(r.json()) == {"id", "patient_id", "ts", "request", "source"}
    assert c.post("/gaze", json=[{"patient_id": 1, "ts": z(now), "x": 0.1, "y": 0.2, "zone": "water",
                                  "dwell_ms": 300, "confidence": 0.9}]).json() == {"inserted": 1}
    assert c.post("/requests", json={"patient_id": 1, "ts": z(now), "request": "coffee"}).status_code == 422
    assert c.get("/requests", params={"patient_id": 1}).json()[0]["request"] == "water"
    assert len(c.get("/gaze/recent", params={"patient_id": 1, "seconds": 60}).json()) == 1


def test_gazepad_style_patient_ids(c):
    now = datetime.now(timezone.utc)
    # GazePad's default id is "p1": it used to be rejected (422) and silently dropped
    r = c.post("/requests", json={"patient_id": "p1", "ts": z(now), "request": "food", "source": "simulated"})
    assert r.status_code == 201 and r.json()["patient_id"] == 1
    # unknown patients are registered on first contact
    r = c.post("/requests", json={"patient_id": "p7", "ts": z(now), "request": "rest"})
    assert r.status_code == 201
    assert any(p["id"] == 7 for p in c.get("/patients").json())


def test_windows_store_exact_b1_message(c):
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
    w = window(1, start)
    r = c.post("/windows", json=w)
    assert r.status_code == 201 and r.json() == {"inserted": 1, "duplicates": 0}
    assert c.post("/windows", json=w).json() == {"inserted": 0, "duplicates": 1}  # retry is harmless
    out = c.get("/windows/latest", params={"patient_id": "USER001"}).json()
    assert list(out["message"].keys()) == B1_FIELDS
    assert out["message"] == w["message"]
    bad = window(1, start + timedelta(seconds=10))
    bad["message"]["Renamed_Field"] = 1
    assert c.post("/windows", json=bad).status_code == 422


def test_sessions_keep_every_parameter(c):
    sid = "test-session-0001"
    params = {"dwell_ms": 1500, "voice": True, "b1": True, "relay": "http://localhost:8100", "screen_cm": 30,
              "dist_cm": 50, "tz": "local", "send_mock_gaze": False, "debug": False}
    r = c.post("/sessions", json={"id": sid, "patient_id": "p1", "mode": "webcam", "params": params,
                                  "client": {"screen_w": 1440}})
    assert r.status_code == 201
    c.post(f"/sessions/{sid}/events", json=[{"type": "calibration_done", "data": {"points": 9}},
                                            {"type": "setting_changed", "data": {"name": "voice", "value": False}}])
    s = c.get(f"/sessions/{sid}").json()
    assert s["params"]["voice"] is False and s["params"]["dwell_ms"] == 1500
    assert [e["type"] for e in s["event_log"]] == ["calibration_done", "setting_changed"]
    assert c.get("/sessions", params={"patient_id": 1}).json()[0]["events"] == 2


def test_adaptive_learning_and_insights(c):
    # patient 2: water at the same time-of-day for 6 days, then ask what they need at that time
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=7)
    for d in range(6):
        for h, label in ((0, "water"), (3, "food"), (6, "rest")):
            t = base + timedelta(days=d, hours=h)
            assert c.post("/requests", json={"patient_id": 2, "ts": z(t), "request": label}).status_code == 201
    probe = base + timedelta(days=6, minutes=2)
    j = c.get("/insights", params={"patient_id": 2, "ts": z(probe)}).json()
    assert j["needs"][0]["label"] == "water"
    assert {"suggestions", "confused", "confusion_score", "alert", "actions", "webcam", "experts", "model"} <= set(j)
    assert len(j["suggestions"]) == 3 and set(j["suggestions"][0]) == {"label", "prob"}
    assert j["model"]["n_learned"] == 18
    assert any("Water" in r for r in j["needs"][0]["reasons"])
    stats = c.get("/model/stats", params={"patient_id": 2}).json()
    assert stats["n"] == 18 and stats["last_20"]["top1"] > 0.5


def test_feedback_lowers_a_handled_need(c):
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=7)
    probe = base + timedelta(days=6, minutes=2)
    before = c.get("/insights", params={"patient_id": 2, "ts": z(probe)}).json()
    p_before = next(n["prob"] for n in before["needs"] if n["label"] == "water")
    assert c.post("/feedback", json={"patient_id": 2, "label": "water", "action": "done",
                                     "ts": z(probe - timedelta(minutes=1))}).status_code == 201
    after = c.get("/insights", params={"patient_id": 2, "ts": z(probe)}).json()
    p_after = next(n["prob"] for n in after["needs"] if n["label"] == "water")
    assert p_after < p_before
    assert after["feedback_applied"][0]["action"] == "done"


def test_webcam_state_reaches_insights(c):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(minutes=40)
    calm = [window(3, start + timedelta(seconds=10 * i), Head_Movement=1.0 + 0.1 * (i % 3)) for i in range(200)]
    c.post("/windows", json=calm)
    t0 = now - timedelta(seconds=80)
    agitated = [window(3, t0 + timedelta(seconds=10 * i), Head_Movement=12.0, Gaze_Stability=0.2) for i in range(6)]
    c.post("/windows", json=agitated)
    j = c.get("/insights", params={"patient_id": 3, "ts": z(now)}).json()
    assert j["webcam"]["available"] and j["webcam"]["personal_baseline"]
    assert j["webcam"]["states"]["distress"] > 0.7
    assert any(a["id"] == "state:distress" for a in j["actions"])


def test_predict_endpoint_unchanged(c):
    r = c.post("/predict", json={"patient_id": 1, "ts": z(datetime.now(timezone.utc))})
    assert r.status_code in (200, 503)  # 503 only if model.joblib is missing
    if r.status_code == 200:
        assert set(r.json()) == {"suggestions", "confused", "confusion_score"}


def test_exports(c):
    for path in ("/export/requests.csv", "/export/windows.csv", "/export/sessions.csv", "/export/evals.csv",
                 "/export/feedback.csv"):
        r = c.get(path)
        assert r.status_code == 200 and r.text.count("\n") >= 1, path
    assert c.get("/export/windows.csv").text.splitlines()[0].endswith(",".join(B1_FIELDS[2:]))


def test_learning_is_fast(c):
    t = time.time()
    c.post("/requests", json={"patient_id": 2, "ts": z(datetime.now(timezone.utc)), "request": "pain"})
    assert time.time() - t < 2.0
