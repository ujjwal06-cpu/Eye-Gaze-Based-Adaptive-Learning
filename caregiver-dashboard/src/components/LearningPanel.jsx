import { Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";
import { META } from "../config.js";
import { pct } from "../utils.js";

const TOP1 = "#3b8fd6";
const TOP3 = "#d97706";
const STATE_TEXT = { distress: "Discomfort", fatigue: "Tiredness", confusion: "Confusion", emergency: "Help pressed" };

function Kpi({ label, value, sub }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-2xl font-bold tabular-nums text-slate-100">{value}</div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

function Tip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-900 px-2 py-1 text-xs shadow-lg">
      <div className="text-slate-400">after {d.i} requests</div>
      <div className="text-slate-100">Top-1 {pct(d.top1)} · Top-3 {pct(d.top3)}</div>
    </div>
  );
}

/** How well the adaptive model is doing for this patient and what it has learned. */
export default function LearningPanel({ statsPoll, insightsPoll }) {
  const st = statsPoll.data;
  const ins = insightsPoll.data;
  const experts = (ins?.experts ?? []).filter((e) => e.weight != null).sort((a, b) => b.weight - a.weight);
  const effects = ins?.model?.personal_model?.state_effects;
  const learned = [];
  if (effects) {
    for (const [s, byLabel] of Object.entries(effects)) {
      for (const [l, v] of Object.entries(byLabel)) if (v >= 0.4) learned.push({ s, l, v });
    }
    learned.sort((a, b) => b.v - a.v);
  }
  const fmt = (a) => (a?.top1 == null ? "-" : pct(a.top1));
  const fmt3 = (a) => (a?.top3 == null ? "-" : pct(a.top3));

  return (
    <section aria-label="Adaptive learning" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Adaptive learning</h2>
        <span className="text-xs text-slate-500">every prediction scored 15 s before the real request</span>
      </div>
      {!st && !statsPoll.error && <p className="text-slate-400">Loading...</p>}
      {!st && statsPoll.error && <p className="text-sm text-amber-300">Unavailable ({statsPoll.error.message}).</p>}
      {st && st.n === 0 && <p className="text-slate-400">No requests learned yet. Accuracy appears after the first requests.</p>}

      {st && st.n > 0 && (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Kpi label="Requests learned" value={st.n} />
            <Kpi label="Top-1, last 20" value={fmt(st.last20)} sub={`top-3 ${fmt3(st.last20)}`} />
            <Kpi label="Top-1, all" value={fmt(st.overall)} sub={`top-3 ${fmt3(st.overall)}`} />
            <Kpi
              label="With webcam"
              value={st.webcam?.n ? fmt(st.webcam) : "-"}
              sub={st.webcam?.n ? `n=${st.webcam.n} · without ${fmt(st.noWebcam)}` : "no webcam data yet"}
            />
          </div>

          {st.rolling.length > 1 && (
            <div className="mt-3 h-40">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={st.rolling} margin={{ top: 6, right: 12, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1f2a44" vertical={false} />
                  <XAxis dataKey="i" type="number" domain={["dataMin", "dataMax"]} stroke="#475569"
                    tick={{ fill: "#8da0c4", fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#475569"
                    tick={{ fill: "#8da0c4", fontSize: 10 }} tickCount={3} />
                  <Tooltip content={<Tip />} cursor={{ stroke: "#64748b", strokeDasharray: "3 3" }} />
                  <Legend verticalAlign="top" height={20} iconType="plainline" wrapperStyle={{ fontSize: 11, color: "#cbd5e1" }} />
                  <Line name="Top-1 (rolling 20)" dataKey="top1" stroke={TOP1} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line name="Top-3 (rolling 20)" dataKey="top3" stroke={TOP3} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {experts.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-xs text-slate-400">How much each source of evidence is trusted for this patient</div>
          <ul className="space-y-1">
            {experts.map((e) => (
              <li key={e.name} className="flex items-center gap-2 text-xs">
                <span className="w-44 shrink-0 truncate text-slate-300">{e.text}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full rounded-full bg-slate-400" style={{ width: pct(e.weight) }} />
                </div>
                <span className="w-10 text-right tabular-nums text-slate-300">{pct(e.weight)}</span>
                <span className="w-20 truncate text-slate-500">{e.awake && e.top ? `→ ${META[e.top]?.text ?? e.top}` : "silent"}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {learned.length > 0 && (
        <p className="mt-3 text-xs text-slate-400">
          Learned for this patient:{" "}
          {learned.slice(0, 3).map((x, i) => (
            <span key={x.s + x.l}>
              {i ? "; " : ""}
              {STATE_TEXT[x.s]} → {META[x.l]?.text}
            </span>
          ))}
        </p>
      )}
    </section>
  );
}
