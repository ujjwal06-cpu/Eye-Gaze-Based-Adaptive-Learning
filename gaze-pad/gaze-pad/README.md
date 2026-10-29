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
