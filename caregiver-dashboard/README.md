# Caregiver Dashboard (Person 4)

React + Vite + Tailwind + Recharts dashboard for caregivers. Polls the FastAPI backend and shows:

- **Suggested now**: top-3 likely requests with probabilities, refreshed every 10 s via `POST /predict`
- **Red "Patient may be confused" banner** (plus a pulsing red card) when `confused` is true
- **Gaze heatmap**: share of gaze samples per tile over the last 5 min, with a white outline on the tile being looked at right now
- **Last 24 hours**: request timeline (one row per label) for the selected patient
- **Latest requests**: live feed, newest first; help and pain are highlighted red

## Run

```bash
npm install
npm run dev        # http://localhost:5174
npm test           # unit tests for parsing / normalisation
npm run build
```

By default it talks to `http://localhost:8000`. Override with `VITE_API_BASE` or `?api=http://host:8000`.

URL params: `?mock=1` (start in mock mode), `?api=...`, `?patient=p1`.

## Mock mode (no backend needed)

- Start with `?mock=1`, or use the header button. If the backend cannot be reached on first load, the dashboard **falls back to mock data automatically** and says so in an amber bar (with a "Retry live" button). A **MOCK DATA** badge is always shown in the header so mock data is never mistaken for real data.
- Three mock patients with distinct daily routines (history is seeded, so charts do not jitter between polls).
- Header controls in mock mode: **Simulate confusion** (gaze starts scattering, the banner appears) and **Add request** (a new request appears in the feed with a NEW tag).

## What it calls

| Call | Used for |
|---|---|
| `GET /patients` | patient selector (every 30 s) |
| `GET /requests?patient_id=&from=&to=` | timeline + feed, last 24 h (every 5 s); `from`/`to` are ISO-8601 UTC with `Z` |
| `GET /gaze/recent?patient_id=&seconds=300` | heatmap (every 2 s) |
| `POST /predict` `{patient_id, ts}` | Suggested now + confusion banner (every 10 s) |

## Response shapes it accepts

The contract did not pin these down, so parsing is deliberately tolerant (see `src/utils.js`):

- **Lists** may be a bare array or wrapped: `{items|patients|requests|events|gaze|data: [...]}`.
- **Patients**: strings, or objects with `id` or `patient_id`, and optional `name`.
- **Requests**: `{patient_id, ts, request, source}` (`label` also accepted for `request`).
- **Gaze**: `{patient_id, ts, x, y, zone, dwell_ms, confidence}`. If `zone` is missing it is derived from `x`,`y`; an explicit `null` is kept (outside the pad).
- **Predict**: `{suggestions:[{label, prob}], confused, confusion_score}`; sorted by `prob`, top 3 kept.
- **Timestamps without a timezone are treated as UTC.** FastAPI/SQLAlchemy often return naive ISO strings; a browser would read those as local time and shift everything by the viewer's UTC offset. Best fix is for the backend to return `Z`/offset timestamps.

If a poll fails, the last good data stays on screen and a "Connection problem, retrying" chip appears in live mode.

## Files

```
src/config.js                  contract constants, poll intervals, zone labels/colours (same order as GazePad)
src/utils.js                   parseTs, normalisers, zoneFromXY, formatting (pure, tested)
src/api.js                     liveApi(base): fetch + normalise
src/mock.js                    mockApi: seeded patients, history, gaze, predict + demo controls
src/hooks.js                   usePoll (resets on patient change, keeps last data on error), useNow
src/App.jsx                    state, polling, mock fallback, layout
src/components/Header.jsx
src/components/ConfusedBanner.jsx
src/components/SuggestionCard.jsx
src/components/GazeHeatmap.jsx
src/components/Timeline.jsx    Recharts scatter, one row per label
src/components/RequestFeed.jsx
src/utils.test.js
```

## Notes

- Polling, not websockets: simplest thing that works with the contract. Worst-case delay for a new request is 5 s.
- The bundle is about 550 kB (Recharts). Fine on a local network; code-split if it ever matters.
- Tested in headless Chromium (browser set to India time) in three setups: mock mode, a stubbed backend returning naive timestamps and wrapped lists, and backend-down fallback. It has not been run against the real FastAPI backend yet.

## Adaptive panels

**Care actions** (with Done / Not needed, sent to `POST /feedback`), **Patient state** (webcam discomfort, tiredness and
confusion against the patient's baseline), **Webcam signals** (last 30 min of B1 windows), and **Adaptive learning**
(out-of-sample accuracy over time, trust per evidence source). "Suggested now" uses `GET /insights` and falls back to
`POST /predict` on an older backend. Mock mode covers every panel.
