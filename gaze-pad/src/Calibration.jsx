import { useEffect, useRef, useState } from "react";
import { CAL_POINTS, CAL_SETTLE_MS, CAL_RECORD_MS, CAL_SAMPLE_EVERY_MS } from "./config.js";

const TOTAL_MS = CAL_SETTLE_MS + CAL_RECORD_MS;

/**
 * 9-point, hands-free calibration. The patient only has to look at each dot:
 * after a short settle time we feed WebGazer the dot's screen position as
 * training data every ~80 ms (the same thing a click does in the classic flow).
 */
export default function Calibration({ onDone, onCancel }) {
  const [started, setStarted] = useState(false);
  const [i, setI] = useState(0);
  const [prog, setProg] = useState(0);
  const cb = useRef({ onDone, onCancel });
  cb.current = { onDone, onCancel };

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") cb.current.onCancel?.();
      if (!started && (e.key === "Enter" || e.key === " ")) setStarted(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [started]);

  useEffect(() => {
    if (!started) return;
    const [fx, fy] = CAL_POINTS[i];
    const px = fx * window.innerWidth;
    const py = fy * window.innerHeight;
    const t0 = performance.now();
    let lastRec = -Infinity;

    const id = setInterval(() => {
      const e = performance.now() - t0;
      setProg(Math.min(1, e / TOTAL_MS));
      if (e >= CAL_SETTLE_MS && e - lastRec >= CAL_SAMPLE_EVERY_MS) {
        lastRec = e;
        try {
          window.webgazer?.recordScreenPosition(px, py, "click");
        } catch {
          /* tracker not ready yet: skip this sample */
        }
      }
      if (e >= TOTAL_MS) {
        clearInterval(id);
        if (i + 1 >= CAL_POINTS.length) cb.current.onDone?.();
        else {
          setProg(0);
          setI(i + 1);
        }
      }
    }, 40);
    return () => clearInterval(id);
  }, [started, i]);

  if (!started) {
    return (
      <div className="cal-overlay">
        <div className="cal-card">
          <h2>Calibration</h2>
          <ol>
            <li>Sit about 50 cm from the screen with your face centred in the camera preview.</li>
            <li>Keep your head as still as you can and look at each dot until it moves.</li>
            <li>There are 9 dots and it takes about 20 seconds.</li>
          </ol>
          <div className="row">
            <button className="btn primary" onClick={() => setStarted(true)}>Start calibration</button>
            <button className="btn" onClick={() => cb.current.onCancel?.()}>Cancel</button>
          </div>
          <p className="hint">Press Enter to start, Esc to cancel.</p>
        </div>
      </div>
    );
  }

  const [fx, fy] = CAL_POINTS[i];
  const recording = prog * TOTAL_MS >= CAL_SETTLE_MS;
  return (
    <div className="cal-overlay run">
      <div className="cal-count">Point {i + 1} of {CAL_POINTS.length}</div>
      <button className="btn cal-cancel" onClick={() => cb.current.onCancel?.()}>Cancel (Esc)</button>
      <div className="cal-dot-wrap" style={{ left: `${fx * 100}%`, top: `${fy * 100}%` }}>
        <div className="cal-shrink" style={{ transform: `translate(-50%, -50%) scale(${1 - 0.7 * prog})` }} />
        <div className={`cal-dot ${recording ? "rec" : ""}`} />
      </div>
    </div>
  );
}
