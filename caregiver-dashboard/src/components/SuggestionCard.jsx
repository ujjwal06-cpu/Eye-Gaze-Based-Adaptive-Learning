import { META, POLL } from "../config.js";
import { pct, timeAgo } from "../utils.js";

export default function SuggestionCard({ poll, now }) {
  const p = poll.data;
  const top = p?.suggestions?.[0];
  const rest = p?.suggestions?.slice(1) ?? [];
  const confused = !!p?.confused;
  const topMeta = top ? META[top.label] : null;

  return (
    <section
      aria-label="Suggested now"
      className={`h-full rounded-2xl border p-6 ${
        confused ? "pulse-red border-red-400 bg-red-950/30" : "border-slate-700 bg-slate-900"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Suggested now{p?.adaptive ? <span className="ml-2 normal-case tracking-normal text-sky-300">adaptive</span> : null}
        </h2>
        <span className="text-xs text-slate-500">
          {poll.updatedAt
            ? `Updated ${timeAgo(poll.updatedAt, now)} · refreshes every ${POLL.predictMs / 1000}s`
            : "Loading..."}
        </span>
      </div>

      {!p && poll.error && (
        <p className="mt-6 text-amber-300">Prediction unavailable ({poll.error.message}). Retrying.</p>
      )}
      {!p && !poll.error && <p className="mt-6 text-slate-400">Loading prediction...</p>}
      {p && !top && <p className="mt-6 text-slate-400">The model returned no suggestions.</p>}

      {top && (
        <>
          <div className="mt-4 flex items-center gap-5">
            <span className="text-7xl leading-none" aria-hidden="true">
              {topMeta?.icon}
            </span>
            <div>
              <div className="text-5xl font-extrabold leading-tight" style={{ color: topMeta?.color }}>
                {topMeta?.text ?? top.label}
              </div>
              <div className="text-lg text-slate-300">{pct(top.prob)} likely</div>
            </div>
          </div>
          {p.needs?.[0]?.reasons?.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm text-slate-300">
              {p.needs[0].reasons.slice(0, 3).map((r) => (
                <li key={r} className="flex gap-2">
                  <span className="text-slate-500" aria-hidden="true">›</span>
                  {r}
                </li>
              ))}
            </ul>
          )}

          <ul className="mt-6 space-y-3">
            {[top, ...rest].map((s) => {
              const m = META[s.label];
              return (
                <li key={s.label} className="flex items-center gap-3">
                  <span className="w-28 shrink-0 text-sm text-slate-200">
                    {m?.icon} {m?.text ?? s.label}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full"
                      style={{ width: pct(Math.min(1, s.prob)), background: m?.color ?? "#94a3b8" }}
                    />
                  </div>
                  <span className="w-12 text-right text-sm tabular-nums text-slate-300">{pct(s.prob)}</span>
                </li>
              );
            })}
          </ul>

          <div className="mt-6 flex items-center gap-3 text-xs text-slate-400">
            <span className="w-28 shrink-0">Confusion score</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
              <div
                className={`h-full rounded-full ${confused ? "bg-red-400" : "bg-emerald-400"}`}
                style={{ width: pct(Math.min(1, p.confusion_score)) }}
              />
            </div>
            <span className="w-12 text-right tabular-nums">{p.confusion_score.toFixed(2)}</span>
          </div>

          {confused && (
            <p className="mt-4 text-sm text-red-200">
              Confusion detected: these suggestions are based on the patient's usual routine at this time of day.
            </p>
          )}
          {poll.error && (
            <p className="mt-3 text-xs text-amber-300">Showing the last prediction (connection problem).</p>
          )}
        </>
      )}
    </section>
  );
}
