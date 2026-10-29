import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { clock } from "../utils.js";

export const TREND_MINUTES = 30;
const LINE = "#3b8fd6";
const METRICS = [
  { key: "Gaze_Stability", text: "Gaze stability", d: 2, domain: [0, 1] },
  { key: "Head_Movement", text: "Head movement (°)", d: 1, domain: [0, "auto"] },
  { key: "Mean_Gaze_Deviation", text: "Gaze deviation (°)", d: 1, domain: [0, "auto"] },
  { key: "Blink_Rate", text: "Blinks per 10 s", d: 0, domain: [0, "auto"] },
];

function Tip({ active, payload, metric }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-900 px-2 py-1 text-xs shadow-lg">
      <div className="text-slate-400">{clock(d.t)}</div>
      <div className="font-semibold text-slate-100 tabular-nums">
        {metric.text}: {Number(d[metric.key]).toFixed(metric.d)}
      </div>
    </div>
  );
}

/** One small chart per webcam signal (B1 window fields), last 30 minutes. */
export default function WebcamTrends({ poll, now }) {
  const rows = (poll.data ?? []).filter((w) => w.t >= now - TREND_MINUTES * 60000);
  const from = now - TREND_MINUTES * 60000;
  return (
    <section aria-label="Webcam trends" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Webcam signals</h2>
        <span className="text-xs text-slate-500">last {TREND_MINUTES} min · {rows.length} windows of 10 s</span>
      </div>
      {poll.data && rows.length === 0 && (
        <p className="py-10 text-center text-slate-400">No webcam windows in the last {TREND_MINUTES} minutes.</p>
      )}
      {poll.error && !poll.data && <p className="text-sm text-amber-300">Could not load windows ({poll.error.message}).</p>}
      {rows.length > 0 && (
        <div className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
          {METRICS.map((m) => {
            const last = rows[rows.length - 1][m.key];
            return (
              <div key={m.key}>
                <div className="flex items-baseline justify-between text-xs">
                  <span className="text-slate-300">{m.text}</span>
                  <span className="font-semibold tabular-nums text-slate-100">{Number(last).toFixed(m.d)}</span>
                </div>
                <div className="h-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rows} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
                      <CartesianGrid stroke="#1f2a44" vertical={false} />
                      <XAxis
                        type="number" dataKey="t" domain={[from, now]} tickFormatter={clock} tickCount={4}
                        stroke="#475569" tick={{ fill: "#8da0c4", fontSize: 10 }}
                      />
                      <YAxis domain={m.domain} stroke="#475569" tick={{ fill: "#8da0c4", fontSize: 10 }} tickCount={3} />
                      <Tooltip content={<Tip metric={m} />} cursor={{ stroke: "#64748b", strokeDasharray: "3 3" }} />
                      <Line type="monotone" dataKey={m.key} stroke={LINE} strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
