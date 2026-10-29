import { useMemo } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { META, TIMELINE_HOURS, ZONES } from "../config.js";
import { clock } from "../utils.js";

const HOUR = 3600000;

function TipContent({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const m = META[d.label];
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm shadow-lg">
      <strong style={{ color: m.color }}>
        {m.icon} {m.text}
      </strong>
      <div className="text-slate-300">{clock(d.t)}</div>
    </div>
  );
}

export default function Timeline({ poll, now }) {
  const from = now - TIMELINE_HOURS * HOUR;
  const requests = poll.data ?? [];

  const series = useMemo(
    () =>
      ZONES.map((z, i) => ({
        ...z,
        data: requests
          .filter((r) => r.request === z.label && r.t >= from && r.t <= now)
          .map((r) => ({ x: r.t, y: i, t: r.t, label: z.label })),
      })),
    [requests, from, now]
  );

  // A tick every 3 hours on local-time boundaries.
  const ticks = useMemo(() => {
    const out = [];
    const d = new Date(from);
    d.setMinutes(0, 0, 0);
    d.setHours(d.getHours() + 1);
    while (d.getTime() <= now) {
      if (d.getHours() % 3 === 0) out.push(d.getTime());
      d.setHours(d.getHours() + 1);
    }
    return out;
  }, [from, now]);

  const count = series.reduce((n, s) => n + s.data.length, 0);

  return (
    <section aria-label="24 hour timeline" className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Last 24 hours</h2>
        <span className="text-xs text-slate-500">{count} requests</span>
      </div>

      <div className="relative h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#1f2a44" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[from, now]}
              ticks={ticks}
              tickFormatter={(v) => clock(v)}
              stroke="#8da0c4"
              tick={{ fill: "#8da0c4", fontSize: 12 }}
              allowDataOverflow
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-0.5, ZONES.length - 0.5]}
              ticks={ZONES.map((_, i) => i)}
              tickFormatter={(v) => (ZONES[v] ? `${ZONES[v].icon} ${ZONES[v].text}` : "")}
              reversed
              width={104}
              stroke="#8da0c4"
              tick={{ fill: "#cbd5e1", fontSize: 12 }}
              interval={0}
            />
            <ZAxis type="number" range={[90, 90]} />
            <Tooltip content={<TipContent />} cursor={{ strokeDasharray: "3 3", stroke: "#475569" }} />
            <ReferenceLine x={now} stroke="#64748b" strokeDasharray="4 4" />
            {series.map((s) => (
              <Scatter key={s.label} name={s.text} data={s.data} fill={s.color} isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>

        {poll.data && count === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-400">
            No requests in the last 24 hours
          </div>
        )}
      </div>
      {poll.error && !poll.data && <p className="mt-2 text-sm text-amber-300">Could not load requests ({poll.error.message}).</p>}
    </section>
  );
}
