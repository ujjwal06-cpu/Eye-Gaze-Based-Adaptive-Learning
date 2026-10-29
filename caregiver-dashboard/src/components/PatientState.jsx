// Webcam-derived patient state: three heuristic states and the signals behind them,
// each compared with this patient's own baseline.
const STATES = [
  { key: "distress", text: "Discomfort", hint: "head movement, unstable gaze, blinking" },
  { key: "fatigue", text: "Tiredness", hint: "long fixations, blinking, little movement" },
  { key: "confusion", text: "Confusion", hint: "wide-ranging, unstable gaze" },
];
const SIGNALS = [
  { key: "head", text: "Head movement", unit: "°", d: 1 },
  { key: "stability", text: "Gaze stability", unit: "", d: 2 },
  { key: "deviation", text: "Gaze deviation", unit: "°", d: 1 },
  { key: "blink", text: "Blinks / 10 s", unit: "", d: 0 },
  { key: "fixation", text: "Avg fixation", unit: " s", d: 2 },
];

function level(v) {
  if (v >= 0.75) return { text: "High", bar: "bg-red-400", ink: "text-red-200", icon: "▲" };
  if (v >= 0.5) return { text: "Raised", bar: "bg-amber-400", ink: "text-amber-200", icon: "●" };
  return { text: "Normal", bar: "bg-emerald-400", ink: "text-emerald-200", icon: "✓" };
}

export default function PatientState({ poll }) {
  const ws = poll.data?.webcam;
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : "-");
  return (
    <section aria-label="Patient state from webcam" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Patient state (webcam)</h2>
        {ws?.available && (
          <span className="text-xs text-slate-500">
            last {ws.n_recent} windows · {ws.personal_baseline ? "vs. personal baseline" : "vs. population defaults"}
          </span>
        )}
      </div>

      {!poll.data && <p className="text-slate-400">Loading...</p>}
      {poll.data && !ws?.available && (
        <p className="text-slate-400">
          No webcam window in the last 90 s
          {Number.isFinite(ws?.last_window_age_s) ? ` (last one ${Math.round(ws.last_window_age_s / 60)} min ago)` : ""}.
          Start GazePad in webcam mode to see the patient's state.
        </p>
      )}

      {ws?.available && (
        <>
          <ul className="space-y-3">
            {STATES.map((s) => {
              const v = ws.states?.[s.key] ?? 0;
              const l = level(v);
              return (
                <li key={s.key}>
                  <div className="mb-1 flex items-baseline justify-between text-sm">
                    <span className="text-slate-200">{s.text}</span>
                    <span className={`text-xs font-semibold ${l.ink}`}>
                      <span aria-hidden="true">{l.icon}</span> {l.text} · {Math.round(v * 100)}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800" title={s.hint}>
                    <div className={`h-full rounded-full ${l.bar}`} style={{ width: `${Math.round(v * 100)}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
          <table className="mt-4 w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500">
                <th className="font-normal">Signal</th>
                <th className="text-right font-normal">Now</th>
                <th className="text-right font-normal">Usual</th>
                <th className="text-right font-normal">Change</th>
              </tr>
            </thead>
            <tbody>
              {SIGNALS.map((g) => {
                const z = ws.z?.[g.key] ?? 0;
                const big = Math.abs(z) >= 1.5;
                return (
                  <tr key={g.key} className="border-t border-slate-800">
                    <td className="py-1 text-slate-300">{g.text}</td>
                    <td className="py-1 text-right tabular-nums text-slate-100">{fmt(ws.now?.[g.key], g.d)}{g.unit}</td>
                    <td className="py-1 text-right tabular-nums text-slate-400">{fmt(ws.baseline?.[g.key], g.d)}{g.unit}</td>
                    <td className={`py-1 text-right tabular-nums ${big ? "font-semibold text-amber-200" : "text-slate-400"}`}>
                      {z > 0.3 ? "↑" : z < -0.3 ? "↓" : "→"} {Math.abs(z).toFixed(1)}σ
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-slate-500">
            Heuristic signals, not a diagnosis. How much they matter for each patient is learned from their requests.
          </p>
        </>
      )}
    </section>
  );
}
