# Eye-gaze adaptive need prediction

A ward simulation for gaze-based adaptive learning. GazePad captures gaze and webcam signals, the backend
stores everything and learns each patient's needs, and the caregiver dashboard shows what each patient is
likely to need now, why, and what to do about it.

```
GazePad (browser) ── POST /gaze, /requests ───────────────▶ FastAPI backend :8000 ◀── caregiver dashboard :5174
        │          ── POST /windows, /sessions (NEW) ──────▶   SQLite app.db            GET /insights, /windows,
        │                                                      adaptive engine          /model/stats, POST /feedback
        └───────── POST /window (UNCHANGED) ──▶ b1-server :8100 ◀── GET /latest (UNCHANGED) ── Team B's B1
```

## What did not change

- **The B1 link is untouched.** `b1-server/` has no edits. GazePad still posts `{message, key, simulated}` to
  `http://localhost:8100/window`, and B1 still polls `GET /latest`, with the same 10 fields
  (`User_ID, Time_Window, Mean_Gaze_Deviation, Max_Gaze_Deviation, Blink_Rate, Avg_Fixation_Duration,
  Max_Fixation_Duration, Head_Movement, Gaze_Stability, Emergency_Button`). `b1Window.js` and
  `windowAggregator.js` are unchanged too, so the numbers are computed exactly as before.
- **The GazePad URL parameters are untouched**: `?patient= &api= &mock= &mockgaze= &b1= &relay= &screencm=
  &distcm= &tz= &debug=` mean the same thing, and none were added. (The two new pad options, Hints and
  Adaptive dwell, are checkboxes saved in the browser.)
- **The original backend endpoints keep their contract**: `/patients`, `/gaze`, `/gaze/recent`,
  `/requests`, `/predict` and `/export/requests.csv`. The old tables are not altered, and an existing
  `app.db` works as it is: the new tables are added on start-up.

## What is new

1. **Everything is stored** (in `backend/app.db`):
   - `gaze_windows`: every 10 s webcam window, which is the exact B1 message plus the session id and valid-slot count
   - `capture_sessions`: every setting selected in GazePad (mode, dwell, voice, gaze dot, camera, B1 on/off,
     relay URL, screen cm, distance cm, tz, hints, adaptive dwell, the raw URL query) and the screen, pad and browser
   - `session_events`: calibration started, done or cancelled, recalibration, setting changes, face lost or
     found, each request (with what was predicted at that moment), and session end
   - `care_feedback`: the caregiver's Done / Not needed on each suggestion
   - `prediction_evals`: for every request, what the model predicted 15 s before it and whether it was right
   - `model_state`: each patient's adaptive model, plus the ward-wide model
2. **Per-patient prediction** (`backend/adaptive/`) from the requests, the webcam windows, live gaze and
   caregiver feedback. It learns from every new request. See `backend/adaptive/README.md`.
3. **Suggestions based on the prediction**: care actions with a priority, plain-language reasons, upcoming
   routine needs, and an alert level (routine, attention or urgent).
4. **Ward simulator** (`backend/sim/`): five virtual patients with different conditions, used to
   evaluate the learning offline, to seed the dashboard, or to stream live patients.

## Run it

```bash
# 1. backend
cd backend
pip install -r requirements.txt          # no new dependencies
python -m sim.ward seed                  # optional: 5 virtual patients with 14 days of history
uvicorn app.main:app --reload --port 8000

# 2. B1 server (unchanged)
cd b1-server && node server.js

# 3. GazePad
cd gaze-pad && npm run dev               # http://localhost:5173/?patient=1

# 4. caregiver dashboard
cd caregiver-dashboard && npm run dev    # http://localhost:5174

# optional: live virtual patients (webcam windows every 10 s, needs every few minutes)
cd backend && python -m sim.ward live --speed 4
```

On the first start, the backend replays the existing requests in time order to train every patient's
model. That takes about 10 s for the current database and runs in the background.

## Simulation results (`python -m sim.ward evaluate --days 21`)

The setup is 5 virtual patients over 21 days, with 990 requests. Each prediction is made 15 s before the
request using only earlier data. A tie counts as a miss.

| top-1 / top-3 | adaptive | adaptive without webcam | routine only | most frequent |
|---|---|---|---|---|
| days 1-7   | 0.68 / 0.87 | 0.61 / 0.79 | 0.56 / 0.78 | 0.24 / 0.70 |
| days 8-14  | 0.77 / 0.94 | 0.69 / 0.89 | 0.65 / 0.90 | 0.23 / 0.74 |
| days 15-21 | 0.78 / 0.94 | 0.70 / 0.87 | 0.69 / 0.88 | 0.22 / 0.72 |

The webcam matters for needs that build up, such as tiredness before Rest (top-1 0.72 with the webcam,
0.20 without) and confusion before Help (0.78 with, 0.49 without). Routine needs are predicted at the same
accuracy with or without it. When one patient's timetable shifted by 90 min, top-1 accuracy fell from 1.00 to 0.43
for the first 3 days and recovered to 0.84 over the next 4.

**These numbers come from simulated patients.** The webcam signatures in `sim/personas.py` are modelling
assumptions. The distress, fatigue and confusion states are heuristics, not diagnoses. Before any clinical
use, the mapping has to be checked on real patients, with ethics approval and clinical oversight.

## Settings (environment variables, backend)

| Variable | Default | Meaning |
|---|---|---|
| `PATIENT_TZ` | server time zone | IANA zone used for time-of-day routines, e.g. `Asia/Kolkata` |
| `ADAPTIVE_LEARN_SOURCES` | `gaze,simulated` | which request sources the model learns from; use `gaze` to ignore mock-mode clicks |
| `ADAPTIVE_STATIC_RF` | `1` | use the existing RandomForest as one expert |
| `ADAPTIVE_CATCH_UP_ON_START` | `1` | learn requests added while the server was down |
| `AUTO_CREATE_PATIENTS` | `1` | register unknown patient ids on first contact (`0` = 404 as before) |

Two fixes came along with this work. GazePad's default patient id `"p1"` used to be rejected by the
backend with a 422, so those requests were silently dropped. `p1`, `1` and `USER001` now all mean
patient 1. The dashboard heatmap also no longer overflows on phone screens.
