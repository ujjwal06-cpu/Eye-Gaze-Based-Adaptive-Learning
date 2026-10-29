import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, DEFAULT_PATIENT_ID, DWELL_MS, ZONES } from "./config.js";
import { Uplink } from "./api.js";
import { useGaze } from "./useGaze.js";
import { setCameraPreview } from "./tracker.js";
import Calibration from "./Calibration.jsx";

const RING_R = 54;
const RING_C = 2 * Math.PI * RING_R;

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
 */
export default function GazePad({
  patientId = DEFAULT_PATIENT_ID,
  apiBase = API_BASE,
  initialMode,
  dwellMs = DWELL_MS,
  voice: voiceDefault = true,
  sendMockGaze = false,
  onRequest,
}) {
  const [patient, setPatient] = useState(patientId);
  const [mode, setMode] = useState(initialMode === "webcam" ? "webcam" : "mock");
  const [phase, setPhase] = useState(
    initialMode === "mock" ? "running" : initialMode === "webcam" ? "loading" : "start"
  );
  const [showDot, setShowDot] = useState(true);
  const [showCam, setShowCam] = useState(false);
  const [voice, setVoice] = useState(voiceDefault);
  const [net, setNet] = useState({ online: null, pendingGaze: 0, pendingRequests: 0 });
  const [toast, setToast] = useState(null);

  const containerRef = useRef(null);
  const uplinkRef = useRef(null);
  const toastTimer = useRef(null);
  const latest = useRef({});
  latest.current = { patient, mode, voice, sendMockGaze, onRequest };

  // Uplink lifecycle (one per backend URL).
  useEffect(() => {
    const up = new Uplink({ apiBase, onStatus: setNet });
    uplinkRef.current = up;
    up.start();
    return () => {
      up.stop();
      uplinkRef.current = null;
    };
  }, [apiBase]);

  const handleSample = useCallback((ev) => {
    const { mode: m, sendMockGaze: smg } = latest.current;
    if (m === "mock" && !smg) return; // keep mouse movement out of real gaze data
    uplinkRef.current?.addGaze(ev);
  }, []);

  const handleSelect = useCallback((label) => {
    const { patient: pid, mode: m, voice: v, onRequest: cbReq } = latest.current;
    const ev = {
      patient_id: pid,
      ts: new Date().toISOString(),
      request: label,
      // Mouse-driven requests are not real gaze, so they are tagged "simulated".
      source: m === "mock" ? "simulated" : "gaze",
    };
    uplinkRef.current?.sendRequest(ev);
    cbReq?.(ev);
    const z = ZONES.find((x) => x.label === label);
    setToast({ id: Date.now(), text: z?.text ?? label, icon: z?.icon });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
    if (v) speakText(`${z?.text ?? label} requested`);
  }, []);

  const gaze = useGaze({
    mode: mode === "mock" ? "mock" : "webgazer",
    enabled: phase !== "start",
    active: phase === "running",
    containerRef,
    patientId: patient,
    dwellMs,
    onSample: handleSample,
    onSelect: handleSelect,
  });

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

  return (
    <div className="screen">
      <header className="bar">
        <div className="bar-left">
          <strong>GazePad</strong>
          <span className="chip">{patient}</span>
          <span className="chip">{mode === "mock" ? "Mock (mouse)" : "Webcam"}</span>
          <span className={`chip ${netClass}`}>{netLabel}</span>
        </div>
        <div className="bar-right">
          <label className="toggle"><input type="checkbox" checked={showDot} onChange={(e) => setShowDot(e.target.checked)} /> Gaze dot</label>
          <label className="toggle"><input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice</label>
          {mode === "webcam" && (
            <>
              <label className="toggle"><input type="checkbox" checked={showCam} onChange={(e) => setShowCam(e.target.checked)} /> Camera</label>
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
          return (
            <div
              key={z.label}
              data-zone={z.label}
              className={`zone${isActive ? " active" : ""}${sent ? " sent" : ""}`}
              style={{ "--c": z.color }}
            >
              <div className="zone-inner">
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
        <Calibration onDone={() => setPhase("running")} onCancel={() => setPhase("running")} />
      )}
    </div>
  );
}
