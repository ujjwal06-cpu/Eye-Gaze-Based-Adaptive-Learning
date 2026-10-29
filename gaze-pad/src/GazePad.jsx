import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, B1, DEFAULT_PATIENT_ID, DWELL_MS, ZONES } from "./config.js";
import { Uplink } from "./api.js";
import { adaptiveDwell, hintLabels } from "./gazeLogic.js";
import { useGaze } from "./useGaze.js";
import { setCameraPreview } from "./tracker.js";
import Calibration from "./Calibration.jsx";
import { createWindowAggregator } from "./windowAggregator.js";

const RING_R = 54;
const RING_C = 2 * Math.PI * RING_R;
const INSIGHTS_MS = 15000; // how often the pad asks the backend what the patient may need

// Per-browser preferences for the two adaptive features (not URL params, so the URL stays as it was)
const loadPref = (k, d) => {
  try {
    const v = localStorage.getItem(`gazepad.${k}`);
    return v === null ? d : v === "1";
  } catch {
    return d;
  }
};
const savePref = (k, v) => {
  try {
    localStorage.setItem(`gazepad.${k}`, v ? "1" : "0");
  } catch {
    /* storage blocked: preference lasts for this page only */
  }
};
const newId = () =>
  globalThis.crypto?.randomUUID?.() ?? `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

function speakText(text) {
  try {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  } catch {
    /* speech is optional feedback */
  }
}

/**
 * <GazePad/>: 6-zone dwell-selection pad driven by webcam gaze (WebGazer) or the mouse (mock mode).
 *
 * props
 *   patientId      string, copied into every event            (default "p1")
 *   apiBase        backend base URL                           (default VITE_API_BASE or http://localhost:8000)
 *   initialMode    "mock" | "webcam" to skip the start screen (optional)
 *   dwellMs        dwell needed to fire a request             (default 1500)
 *   voice          speak the selected request aloud           (default true)
 *   sendMockGaze   also POST mouse "gaze" to /gaze in mock mode (default false, see README)
 *   onRequest(ev)  called with each request event that was queued
 *   b1             send 10 s feature windows to the B1 relay        (default true)
 *   relayBase      B1 relay URL                                      (default VITE_B1_RELAY or http://localhost:8100)
 *   screenWidthCm, viewDistCm, tz   inputs for the gaze-deviation maths (see README)
 *   debug          show a live blink / head / window readout
 *
 * Adaptive additions (no new props or URL params):
 *   - every session's selected settings, calibration and setting changes are stored in the backend
 *   - every 10 s window goes to the B1 server exactly as before AND is stored in the backend
 *   - "Hints" marks the tiles the adaptive model expects; "Adaptive dwell" makes the most likely
 *     tile need less dwell (opt-in, never below 1 s)
 */
export default function GazePad({
  patientId = DEFAULT_PATIENT_ID,
  apiBase = API_BASE,
  initialMode,
  dwellMs = DWELL_MS,
  voice: voiceDefault = true,
  sendMockGaze = false,
  onRequest,
  b1 = true,
  relayBase = B1.relayBase,
  screenWidthCm = B1.screenWidthCm,
  viewDistCm = B1.viewDistCm,
  tz = B1.tz,
  debug = false,
}) {
  const [patient, setPatient] = useState(patientId);
  const [mode, setMode] = useState(initialMode === "webcam" ? "webcam" : "mock");
  const [phase, setPhase] = useState(
    initialMode === "mock" ? "running" : initialMode === "webcam" ? "loading" : "start"
  );
  const [showDot, setShowDot] = useState(true);
  const [showCam, setShowCam] = useState(false);
  const [voice, setVoice] = useState(voiceDefault);
  const [net, setNet] = useState({
    online: null, pendingGaze: 0, pendingRequests: 0, pendingWindows: 0, relayOnline: null, windowsSent: 0,
  });
  const [win, setWin] = useState({ made: 0, skipped: 0, last: null, lastSkip: null });
  const [toast, setToast] = useState(null);
  const [hints, setHints] = useState(() => loadPref("hints", true));
  const [adaptive, setAdaptive] = useState(() => loadPref("adaptiveDwell", false));
  const [predictions, setPredictions] = useState([]);
  const sessionRef = useRef(null);

  const containerRef = useRef(null);
  const uplinkRef = useRef(null);
  const toastTimer = useRef(null);
  const aggRef = useRef(null);
  const latest = useRef({});
  latest.current = { patient, mode, voice, sendMockGaze, onRequest, screenWidthCm, viewDistCm, tz, hints, adaptive, predictions };

  // Uplink lifecycle (one per backend URL).
  useEffect(() => {
    const up = new Uplink({ apiBase, relayBase: b1 ? relayBase : null, onStatus: setNet });
    uplinkRef.current = up;
    up.start();
    return () => {
      up.stop();
      uplinkRef.current = null;
    };
  }, [apiBase, relayBase, b1]);

  // B1 windows: one spec-shaped message per completed 10 s wall-clock window.
  // Computed even with b1 off, so the backend still stores the webcam parameters.
  useEffect(() => {
    const agg = createWindowAggregator({
      minValid: B1.minValid,
      getContext: () => {
        const L = latest.current;
        const rect = containerRef.current?.getBoundingClientRect();
        return {
          patientId: L.patient,
          padW: rect?.width || window.innerWidth,
          padH: rect?.height || window.innerHeight,
          cmPerPx: L.screenWidthCm / (window.screen?.width || window.innerWidth),
          viewDistCm: L.viewDistCm,
          fixationDeg: B1.fixationDeg,
          tz: L.tz,
        };
      },
      onWindow: (w) => {
        const env = { message: w.message, key: w.key, simulated: latest.current.mode === "mock" };
        uplinkRef.current?.sendWindow(env); // B1 server: unchanged URL and payload
        uplinkRef.current?.storeWindow(env, { valid: w.valid, session_id: sessionRef.current });
        setWin((s) => ({ ...s, made: s.made + 1, last: w.message }));
      },
      onSkip: ({ start, valid }) => setWin((s) => ({ ...s, skipped: s.skipped + 1, lastSkip: { start, valid } })),
    });
    aggRef.current = agg;
    const id = setInterval(() => agg.tick(Date.now()), 250);
    return () => {
      clearInterval(id);
      aggRef.current = null;
    };
  }, [b1]);

  const handleWindowSample = useCallback((s) => aggRef.current?.addSample(s), []);
  const handleBlink = useCallback((t) => aggRef.current?.addBlink(t), []);

  const handleSample = useCallback((ev) => {
    const { mode: m, sendMockGaze: smg } = latest.current;
    if (m === "mock" && !smg) return; // keep mouse movement out of real gaze data
    uplinkRef.current?.addGaze(ev);
  }, []);

  const handleSelect = useCallback((label) => {
    const { patient: pid, mode: m, voice: v, onRequest: cbReq, hints: h, adaptive: a, predictions: pr } = latest.current;
    if (label === B1.emergencyLabel) aggRef.current?.addEmergency(Date.now());
    const ev = {
      patient_id: pid,
      ts: new Date().toISOString(),
      request: label,
      // Mouse-driven requests are not real gaze, so they are tagged "simulated".
      source: m === "mock" ? "simulated" : "gaze",
    };
    uplinkRef.current?.sendRequest(ev);
    uplinkRef.current?.sessionEvent(sessionRef.current, "request", {
      request: label,
      predicted_top: pr[0]?.label ?? null,
      predicted_prob: pr.find((p) => p.label === label)?.prob ?? null,
      hinted: h && hintLabels(pr).includes(label),
      adaptive_dwell: a,
    });
    cbReq?.(ev);
    // the model has just learned from this request: refresh the hints soon
    setTimeout(() => refreshRef.current?.(), 2500);
    const z = ZONES.find((x) => x.label === label);
    setToast({ id: Date.now(), text: z?.text ?? label, icon: z?.icon });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
    if (v) speakText(`${z?.text ?? label} requested`);
  }, []);

  const refreshRef = useRef(null);
  const dwellFor = useMemo(
    () => (adaptive ? adaptiveDwell(dwellMs, predictions) : null),
    [adaptive, dwellMs, predictions]
  );

  const gaze = useGaze({
    mode: mode === "mock" ? "mock" : "webgazer",
    enabled: phase !== "start",
    active: phase === "running",
    containerRef,
    patientId: patient,
    dwellMs,
    onSample: handleSample,
    onSelect: handleSelect,
    onWindowSample: handleWindowSample,
    onBlink: handleBlink,
    dwellFor,
    debug,
  });

  // ---- session: store every selected parameter, and what happens during the session ----
  const sessionOn = phase !== "start" && phase !== "error";
  const params = {
    dwell_ms: dwellMs, voice, show_dot: showDot, show_camera: showCam, send_mock_gaze: sendMockGaze,
    b1, relay: b1 ? relayBase : null, api: apiBase, screen_cm: screenWidthCm, dist_cm: viewDistCm, tz, debug,
    hints, adaptive_dwell: adaptive, emergency_label: B1.emergencyLabel, min_valid: B1.minValid,
    fixation_deg: B1.fixationDeg, initial_mode: initialMode ?? null,
    url_params: typeof window !== "undefined" ? window.location.search : "",
  };
  const paramsRef = useRef(params);
  paramsRef.current = params;

  useEffect(() => {
    if (!sessionOn) return;
    const up = uplinkRef.current;
    const id = newId();
    sessionRef.current = id;
    const rect = containerRef.current?.getBoundingClientRect();
    up?.startSession({
      id, patient_id: patient, app: "gazepad", mode, started_at: new Date().toISOString(),
      params: paramsRef.current,
      client: {
        user_agent: navigator.userAgent, language: navigator.language,
        screen_w: window.screen?.width, screen_h: window.screen?.height, dpr: window.devicePixelRatio,
        viewport_w: window.innerWidth, viewport_h: window.innerHeight,
        pad_w: rect ? Math.round(rect.width) : null, pad_h: rect ? Math.round(rect.height) : null,
        tz_offset_min: -new Date().getTimezoneOffset(),
      },
    });
    const onClose = () => {
      up?.sessionEvent(id, "ended", { reason: "page_closed" });
      up?.flush();
    };
    window.addEventListener("pagehide", onClose);
    return () => {
      window.removeEventListener("pagehide", onClose);
      up?.sessionEvent(id, "ended", { reason: "mode_or_patient_changed" });
      sessionRef.current = null;
    };
    // a new session whenever the mode, patient or backend changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionOn, mode, patient, apiBase]);

  const logEvent = useCallback((type, data) => uplinkRef.current?.sessionEvent(sessionRef.current, type, data), []);
  const changeSetting = (name, value, setter, pref) => {
    setter(value);
    if (pref) savePref(pref, value);
    logEvent("setting_changed", { name, value });
  };

  useEffect(() => {
    if (phase === "calibrating") logEvent("calibration_started", { points: 9 });
  }, [phase, logEvent]);

  // face lost / found (webcam), at most one event per 5 s
  const faceAt = useRef(0);
  useEffect(() => {
    if (mode !== "webcam" || phase !== "running") return;
    if (Date.now() - faceAt.current < 5000) return;
    faceAt.current = Date.now();
    logEvent(gaze.tracking ? "face_found" : "face_lost", {});
  }, [gaze.tracking, mode, phase, logEvent]);

  // ---- adaptive predictions for this patient (hints and adaptive dwell) ----
  useEffect(() => {
    if (phase !== "running" || (!hints && !adaptive)) {
      setPredictions([]);
      return;
    }
    let alive = true;
    const load = async () => {
      const j = await uplinkRef.current?.insights(patient);
      if (alive && j?.needs) setPredictions(j.needs.map((n) => ({ label: n.label, prob: n.prob })));
    };
    refreshRef.current = load;
    load();
    const id = setInterval(load, INSIGHTS_MS);
    return () => {
      alive = false;
      clearInterval(id);
      refreshRef.current = null;
    };
  }, [phase, hints, adaptive, patient, apiBase]);
  const hinted = hints ? hintLabels(predictions) : [];

  // Webcam ready -> calibrate. Webcam failed -> error screen.
  useEffect(() => {
    if (mode !== "webcam") return;
    if (phase === "loading" && gaze.status === "ready") setPhase("calibrating");
    if ((phase === "loading" || phase === "running") && gaze.status === "error") setPhase("error");
  }, [mode, phase, gaze.status]);

  // Camera preview: always on while calibrating, otherwise follows the toggle.
  useEffect(() => {
    if (mode === "webcam" && gaze.status === "ready") {
      setCameraPreview(phase === "calibrating" || showCam);
    }
  }, [mode, phase, showCam, gaze.status]);

  useEffect(() => () => clearTimeout(toastTimer.current), []);

  const startWebcam = () => {
    setMode("webcam");
    setPhase("loading");
  };
  const startMock = () => {
    setMode("mock");
    setPhase("running");
  };
  const recalibrate = async () => {
    logEvent("recalibrate", {});
    try {
      await window.webgazer?.clearData?.();
    } catch {
      /* nothing to clear */
    }
    setPhase("calibrating");
  };

  // ---------------------------------------------------------------- start screen
  if (phase === "start") {
    return (
      <div className="screen center">
        <div className="card">
          <h1>GazePad</h1>
          <p className="sub">Look at a tile for 1.5 seconds to send a request.</p>
          <label className="field">
            Patient ID
            <input value={patient} onChange={(e) => setPatient(e.target.value.trim())} spellCheck={false} />
          </label>
          <div className="stack">
            <button className="btn primary big" disabled={!patient} onClick={startWebcam}>
              Use webcam eye tracking
            </button>
            <button className="btn big" disabled={!patient} onClick={startMock}>
              Mock mode (mouse = gaze)
            </button>
          </div>
          <p className="hint">Mock mode needs no camera. Its requests are saved with source "simulated".</p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------- error screen
  if (phase === "error") {
    return (
      <div className="screen center">
        <div className="card">
          <h1>Camera not available</h1>
          <p className="sub">{gaze.error || "The eye tracker could not start."}</p>
          <div className="stack">
            <button className="btn primary big" onClick={startWebcam}>Try again</button>
            <button className="btn big" onClick={startMock}>Use mock mode instead</button>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------- pad
  const running = phase === "running";
  const netLabel =
    net.online === false
      ? `Offline, ${net.pendingRequests} request${net.pendingRequests === 1 ? "" : "s"} queued`
      : net.online
      ? "Connected"
      : "Waiting for backend";
  const netClass = net.online === false ? "bad" : net.online ? "ok" : "";
  const b1Label = !b1
    ? null
    : net.relayOnline === false
    ? `B1 server offline, ${net.pendingWindows} queued`
    : net.relayOnline
    ? `B1: ${net.windowsSent} sent`
    : "B1: waiting";
  const b1Class = net.relayOnline === false ? "bad" : net.relayOnline ? "ok" : "";
  const topPred = predictions[0];
  const fi = gaze.faceInfo;

  return (
    <div className="screen">
      <header className="bar">
        <div className="bar-left">
          <strong>GazePad</strong>
          <span className="chip">{patient}</span>
          <span className="chip">{mode === "mock" ? "Mock (mouse)" : "Webcam"}</span>
          <span className={`chip ${netClass}`}>{netLabel}</span>
          {b1Label && <span className={`chip ${b1Class}`}>{b1Label}</span>}
          {hints && topPred && (
            <span className="chip hint" title="Adaptive model: most likely need right now">
              Likely: {ZONES.find((z) => z.label === topPred.label)?.text} {Math.round(topPred.prob * 100)}%
            </span>
          )}
        </div>
        <div className="bar-right">
          <label className="toggle"><input type="checkbox" checked={showDot} onChange={(e) => changeSetting("show_dot", e.target.checked, setShowDot)} /> Gaze dot</label>
          <label className="toggle"><input type="checkbox" checked={voice} onChange={(e) => changeSetting("voice", e.target.checked, setVoice)} /> Voice</label>
          <label className="toggle" title="Mark the tiles the adaptive model expects the patient to need">
            <input type="checkbox" checked={hints} onChange={(e) => changeSetting("hints", e.target.checked, setHints, "hints")} /> Hints
          </label>
          <label className="toggle" title="The most likely tile needs less dwell (never below 1 s)">
            <input type="checkbox" checked={adaptive} onChange={(e) => changeSetting("adaptive_dwell", e.target.checked, setAdaptive, "adaptiveDwell")} /> Adaptive dwell
          </label>
          {mode === "webcam" && (
            <>
              <label className="toggle"><input type="checkbox" checked={showCam} onChange={(e) => changeSetting("show_camera", e.target.checked, setShowCam)} /> Camera</label>
              <button className="btn small" onClick={recalibrate}>Recalibrate</button>
            </>
          )}
          <button className="btn small" onClick={() => setPhase("start")}>Change mode</button>
        </div>
      </header>

      <main className="pad" ref={containerRef}>
        {ZONES.map((z) => {
          const isActive = running && gaze.zone === z.label;
          const sent = isActive && !gaze.armed;
          const progress = isActive ? gaze.progress : 0;
          const hint = running && hinted.includes(z.label);
          const fast = running && dwellFor && dwellFor(z.label) < dwellMs;
          return (
            <div
              key={z.label}
              data-zone={z.label}
              className={`zone${isActive ? " active" : ""}${sent ? " sent" : ""}${hint ? " hinted" : ""}`}
              style={{ "--c": z.color }}
            >
              <div className="zone-inner">
                {hint && (
                  <span className="hint-badge" title="The adaptive model expects this need now">
                    Likely{fast ? " · quick" : ""}
                  </span>
                )}
                <div className="ringwrap">
                  <svg className="ring" viewBox="0 0 120 120" aria-hidden="true">
                    <circle className="ring-bg" cx="60" cy="60" r={RING_R} />
                    <circle
                      className="ring-fg"
                      cx="60" cy="60" r={RING_R}
                      strokeDasharray={RING_C}
                      strokeDashoffset={RING_C * (1 - progress)}
                      transform="rotate(-90 60 60)"
                    />
                  </svg>
                  <span className="icon">{z.icon}</span>
                </div>
                <span className="label">{z.text}</span>
              </div>
            </div>
          );
        })}

        {running && showDot && gaze.tracking && (
          <div className="gazedot" style={{ left: `${gaze.x * 100}%`, top: `${gaze.y * 100}%` }} />
        )}

        {running && mode === "webcam" && !gaze.tracking && (
          <div className="banner">Can't see the eyes. Please face the camera.</div>
        )}

        {debug && (
          <pre className="debughud">
{`window: ${win.made} made, ${win.skipped} skipped${win.lastSkip ? ` (last had ${win.lastSkip.valid}/100 valid)` : ""}
${mode === "webcam" && fi
  ? `face: ${fi.faceSeen ? "seen" : "LOST"}  EAR ${fi.ear?.toFixed(3) ?? "-"} (open baseline ${fi.baseline?.toFixed(3) ?? "-"})  blinks ${fi.blinks}  head step ${fi.headStep.toFixed(2)} deg`
  : "face: n/a (mock mode)"}
last: ${win.last ? JSON.stringify(win.last, null, 1) : "-"}`}
          </pre>
        )}

        {toast && (
          <div className="toast" key={toast.id}>
            <span>{toast.icon}</span> {toast.text} requested
          </div>
        )}
      </main>

      {phase === "loading" && (
        <div className="cal-overlay">
          <div className="cal-card"><h2>Starting camera...</h2><p className="hint">Allow camera access when the browser asks.</p></div>
        </div>
      )}

      {phase === "calibrating" && (
        <Calibration
          onDone={() => {
            logEvent("calibration_done", { points: 9 });
            setPhase("running");
          }}
          onCancel={() => {
            logEvent("calibration_cancelled", {});
            setPhase("running");
          }}
        />
      )}
    </div>
  );
}
