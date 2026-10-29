# GazePad (Person 1: gaze capture)

React + Vite module that captures patient gaze, turns it into requests (water, food, help, pain, bathroom, rest) by dwell selection, and ships both to the FastAPI backend.

## Run

```bash
npm install
npm run dev          # copies webgazer.js into public/, then starts Vite on :5173
npm test             # unit tests for zone mapping + dwell logic
```

Open http://localhost:5173 and pick **Mock mode** (mouse = gaze, no camera) or **Webcam**.

URL params: `?patient=p1&api=http://localhost:8000&mock=1&mockgaze=1`
Env: `VITE_API_BASE` sets the default backend URL (default `http://localhost:8000`).

## Use it as a component

```jsx
import GazePad from "./GazePad.jsx";
<GazePad patientId="p1" apiBase="http://localhost:8000" />
```

Props: `patientId`, `apiBase`, `initialMode` ("mock" | "webcam"), `dwellMs` (1500), `voice` (true), `sendMockGaze` (false), `onRequest(ev)`.

Hook for other UIs: `useGaze({ mode, enabled, active, containerRef, patientId, onSample, onSelect })` returns `{ x, y, zone, dwellMs }` plus `progress, armed, confidence, tracking, status, error`.

## What it sends

- `POST /gaze`: a JSON array of gaze events, flushed every 1 s (10 events/s):
  `{patient_id, ts, x, y, zone, dwell_ms, confidence}`
- `POST /requests`: one request event per completed dwell:
  `{patient_id, ts, request, source}`

Field notes (so the other modules agree):

- `x`, `y` are 0-1 relative to the 6-tile pad, clamped. `zone` is the *instantaneous* zone of the smoothed gaze (null if outside the pad), so fast scattering stays visible for the entropy / zone-switch features.
- `dwell_ms` is how long the gaze has stayed in the current zone (0 if it just moved).
- `confidence` is a jitter heuristic (1 = steady), not a tracker-reported score.
- `source` is `"gaze"` for webcam requests and `"simulated"` for mouse requests in mock mode.
- In mock mode, gaze events are **not** sent unless `sendMockGaze` / `?mockgaze=1` is set, because mouse movement would pollute real gaze data (the gaze event has no `source` field).

## Behaviour

- Zones tile the pad with no gaps: row-major 3x2 = water, food, help / pain, bathroom, rest.
- A zone counts only after the gaze stays in it for 250 ms (border jitter does not reset the dwell). 1.5 s of dwell fires the request once; the patient must look away before the same tile can fire again.
- If the backend is down, gaze is buffered (last 5000 events) and requests are queued in localStorage and retried until delivered. The top bar shows Offline / Connected.
- Calibration is hands-free: 9 dots, the patient just looks at each one (about 20 s). Use **Recalibrate** if the patient moves or the camera is repositioned.
- Spoken confirmation ("Water requested") can be turned off with the Voice toggle.

## Files

```
src/config.js        shared contract constants, timings, calibration points
src/gazeLogic.js     zoneFromXY + DwellTracker (pure, tested)
src/useGaze.js       useGaze() hook: mouse or WebGazer source, smoothing, dwell, sampling
src/tracker.js       loads /webgazer.js, camera preview helpers
src/api.js           Uplink: gaze batching + persistent request queue
src/Calibration.jsx  9-point calibration overlay
src/GazePad.jsx      the component (start screen, pad, controls)
src/gazeLogic.test.js
```

## Known limits

- Webcam gaze (WebGazer) is typically accurate to a few cm on screen; six big tiles are about the finest layout it supports reliably. Good lighting and a still head matter more than anything else.
- Webcam mode could not be tested in the build sandbox (no camera). Mock mode, dwell logic, batching and the build were tested.

## Team B (B1) integration

GazePad also computes the 10 features B1 needs, per 10-second wall-clock window, and sends each
one to the `b1-server` (`POST http://localhost:8100/window`). Team B's B1 then polls that server's `GET /latest`. Start the server first (see `b1-server/README.md`). The old Python `b1-relay` is no longer needed.

URL params: `&b1=0` turn off, `&relay=URL`, `&screencm=30` (screen width in cm), `&distcm=50`
(eye-to-screen distance), `&tz=utc` (Time_Window clock, default local), `&debug=1` (live readout of
blinks, head movement and the last message).

Definitions (Team B's updated spec, `B1_Data_Requirements_4.pdf`):
- Gaze deviation = angle between the gaze point and ONE fixed reference point, the pad (screen) centre. Uses `screencm` / `distcm`.
- Gaze_Stability = 1 / (1 + std of that deviation). Fixation = gaze staying within 2 degrees.
- Blink = eye-aspect-ratio dip below 70% of the person's open-eye baseline for 60 ms to 1 s, then reopening.
- Head_Movement = largest sqrt(dyaw^2 + dpitch^2 + droll^2) between consecutive samples, in degrees.
- Emergency_Button = number of "help" selections. User_ID `1` becomes `USER001`.
- The first partial window and windows with under 50 of 100 gaze samples are skipped (never published as zeros).
- The browser tab must stay in the foreground; background tabs throttle timers.

Note: B1's normal values (mean deviation 2-5, max 6-10 degrees) assume a person looking near the screen centre. Looking at the outer tiles
gives larger deviations, so check what your real windows look like with `?debug=1`.

Blink and head numbers have only been tested on synthetic data, not on a real face: check them with `?debug=1`.

## Adaptive additions (URL params and the B1 payload are unchanged)

- Every 10 s window still goes to the B1 server exactly as before. A copy goes to the backend (`POST /windows`) as well.
- Each session's selected settings (`POST /sessions`) and its events (calibration, setting changes, face lost/found, requests) are stored.
  They are queued in localStorage while the backend is offline. An older backend without these endpoints just ignores them.
- **Hints** (on by default): a "Likely" badge marks the tiles the adaptive model expects (`GET /insights`, every 15 s).
- **Adaptive dwell** (off by default): the most likely tile needs 25% less dwell, never below 1 s. Other tiles keep the normal dwell.
- Both options are checkboxes saved in the browser, not URL params. Windows are computed even with `b1=0`, so they can still be stored.
