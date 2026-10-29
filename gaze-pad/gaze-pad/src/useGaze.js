import { useEffect, useRef, useState } from "react";
import { DWELL_MS, TICK_MS, SAMPLE_MS, STALE_MS } from "./config.js";
import { DwellTracker, zoneFromXY, clamp01 } from "./gazeLogic.js";
import { loadWebGazer, quietWebGazer } from "./tracker.js";

const NO_RAW = { px: 0, py: 0, t: 0 };
const r4 = (v) => Math.round(v * 10000) / 10000;
const r2 = (v) => Math.round(v * 100) / 100;

/**
 * useGaze: turns a raw gaze source into pad-relative coordinates, a zone and a dwell time.
 *
 * Returns { x, y, zone, dwellMs } (x, y normalised 0-1 relative to containerRef) plus
 * { progress, armed, confidence, tracking, status, error } for the UI.
 *
 * options
 *   mode          "mock" (mouse = gaze) | "webgazer" (webcam)
 *   enabled       turn the gaze source on (webcam permission is requested when mode is "webgazer")
 *   active        process gaze, fire selections and emit samples (false during calibration)
 *   containerRef  ref of the pad element; coordinates are normalised to its bounding box
 *   patientId     copied into every gaze event
 *   dwellMs       dwell needed to fire a selection
 *   onSample(ev)  called every SAMPLE_MS with a contract-shaped gaze event
 *   onSelect(label, {x, y})  called once when dwell completes on a zone
 */
export function useGaze({
  mode = "mock",
  enabled = true,
  active = true,
  containerRef,
  patientId,
  dwellMs = DWELL_MS,
  onSample,
  onSelect,
} = {}) {
  const [state, setState] = useState({
    x: 0.5, y: 0.5, zone: null, dwellMs: 0, progress: 0, armed: true, confidence: 0, tracking: false,
  });
  // status is keyed by the source it belongs to, so a stale "ready" from a previous
  // mode can never be read right after switching modes.
  const key = enabled ? mode : "off";
  const [st, setSt] = useState({ key: "off", status: "idle", error: null });
  const fallback = !enabled ? "idle" : mode === "mock" ? "ready" : "loading";
  const status = st.key === key ? st.status : fallback;
  const error = st.key === key ? st.error : null;

  const raw = useRef(NO_RAW); // latest raw gaze in viewport pixels
  const cbs = useRef({});
  cbs.current = { onSample, onSelect, patientId };

  // --- source: mouse or WebGazer -------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    if (mode === "mock") {
      const onMove = (e) => {
        raw.current = { px: e.clientX, py: e.clientY, t: performance.now() };
      };
      window.addEventListener("pointermove", onMove);
      return () => {
        window.removeEventListener("pointermove", onMove);
        raw.current = NO_RAW;
      };
    }

    let cancelled = false;
    let wg = null;
    (async () => {
      try {
        wg = await loadWebGazer();
        wg.setRegression?.("ridge");
        wg.saveDataAcrossSessions?.(false); // calibrate fresh every session
        // Where WebGazer 3.x loads its MediaPipe face mesh files from (copied by scripts/copy-webgazer.mjs).
        if (wg.params) wg.params.faceMeshSolutionPath = `${import.meta.env.BASE_URL}mediapipe/face_mesh`;
        wg.setGazeListener((d) => {
          if (d && Number.isFinite(d.x) && Number.isFinite(d.y)) {
            raw.current = { px: d.x, py: d.y, t: performance.now() };
          }
        });
        await wg.begin();
        if (cancelled) {
          wg.end?.();
          return;
        }
        quietWebGazer(wg);
        setSt({ key, status: "ready", error: null });
      } catch (e) {
        if (cancelled) return;
        console.error("[GazePad] WebGazer failed to start:", e?.stack || e);
        const msg =
          e?.name === "NotAllowedError"
            ? "Camera permission was denied."
            : e?.name === "NotFoundError"
            ? "No camera was found."
            : e?.message || String(e);
        setSt({ key, status: "error", error: msg });
      }
    })();

    return () => {
      cancelled = true;
      raw.current = NO_RAW;
      try {
        wg?.clearGazeListener?.();
        wg?.end?.();
      } catch {
        /* already stopped */
      }
    };
  }, [enabled, mode, key]);

  // --- processing loop: smoothing, zone, dwell, selection, sampling --------------
  useEffect(() => {
    if (!enabled || !active || status !== "ready") return;

    const tracker = new DwellTracker({ dwellMs });
    const alpha = mode === "mock" ? 0.6 : 0.25; // EMA smoothing; webcam gaze is noisier
    let sx = null;
    let sy = null;
    let lastSample = 0;

    const id = setInterval(() => {
      const now = performance.now();
      const el = containerRef?.current;
      const rect = el
        ? el.getBoundingClientRect()
        : { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
      const r = raw.current;
      // A resting mouse is still "looking" at that spot; a webcam that stops reporting is not.
      const fresh =
        r.t > 0 && rect.width > 0 && rect.height > 0 && (mode === "mock" || now - r.t < STALE_MS);

      if (!fresh) {
        sx = sy = null;
        const res = tracker.update(null, now);
        setState((s) =>
          !s.tracking && s.zone === res.zone
            ? s
            : { ...s, zone: res.zone, dwellMs: 0, progress: 0, armed: res.armed, confidence: 0, tracking: false }
        );
        return;
      }

      const nx = (r.px - rect.left) / rect.width;
      const ny = (r.py - rect.top) / rect.height;
      if (sx === null) {
        sx = nx;
        sy = ny;
      } else {
        sx += alpha * (nx - sx);
        sy += alpha * (ny - sy);
      }
      // Confidence heuristic: how far the raw point is from the smoothed one (jitter).
      const confidence = clamp01(1 - Math.hypot(nx - sx, ny - sy) / 0.25);

      const instZone = zoneFromXY(sx, sy);
      const res = tracker.update(instZone, now);
      const cx = clamp01(sx);
      const cy = clamp01(sy);

      if (res.fired) cbs.current.onSelect?.(res.fired, { x: cx, y: cy });

      setState({
        x: cx, y: cy, zone: res.zone, dwellMs: res.dwellMs, progress: res.progress,
        armed: res.armed, confidence, tracking: true,
      });

      if (now - lastSample >= SAMPLE_MS) {
        lastSample = now;
        cbs.current.onSample?.({
          patient_id: cbs.current.patientId,
          ts: new Date().toISOString(),
          x: r4(cx),
          y: r4(cy),
          zone: instZone, // instantaneous zone: keeps fast scattering visible for confusion features
          dwell_ms: instZone && instZone === res.zone ? Math.round(res.dwellMs) : 0,
          confidence: r2(confidence),
        });
      }
    }, TICK_MS);

    return () => clearInterval(id);
  }, [enabled, active, status, dwellMs, mode, containerRef]);

  return { ...state, status, error };
}
