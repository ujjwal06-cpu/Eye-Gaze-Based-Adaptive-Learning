import { FEED_LIMIT, META, URGENT } from "../config.js";
import { clock, timeAgo } from "../utils.js";

export default function RequestFeed({ poll, now }) {
  const rows = [...(poll.data ?? [])].sort((a, b) => b.t - a.t).slice(0, FEED_LIMIT);

  return (
    <section aria-label="Latest requests" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">Latest requests</h2>

      {poll.loading && !poll.data && <p className="text-slate-400">Loading...</p>}
      {poll.error && !poll.data && <p className="text-amber-300">Could not load requests ({poll.error.message}).</p>}
      {poll.data && rows.length === 0 && <p className="text-slate-400">No requests in the last 24 hours.</p>}

      <ul className="space-y-2">
        {rows.map((r, i) => {
          const m = META[r.request];
          const urgent = URGENT.has(r.request);
          const fresh = now - r.t < 60000;
          return (
            <li
              key={`${r.t}-${r.request}-${i}`}
              className={`flex items-center gap-3 rounded-xl border px-3 py-2 ${
                urgent ? "border-red-500/60 bg-red-500/10" : "border-slate-800 bg-slate-950/40"
              }`}
            >
              <span className="text-2xl" aria-hidden="true">
                {m?.icon ?? "❓"}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold" style={{ color: m?.color }}>
                    {m?.text ?? r.request}
                  </span>
                  {fresh && (
                    <span className="rounded bg-sky-500/20 px-1.5 text-[10px] font-bold text-sky-300">NEW</span>
                  )}
                  {r.source === "simulated" && (
                    <span className="rounded bg-slate-700 px-1.5 text-[10px] text-slate-300">sim</span>
                  )}
                </div>
                <div className="text-xs text-slate-400">{timeAgo(r.t, now)}</div>
              </div>
              <span className="text-sm tabular-nums text-slate-300">{clock(r.t)}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
