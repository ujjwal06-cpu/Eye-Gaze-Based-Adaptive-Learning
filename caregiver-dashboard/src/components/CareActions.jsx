import { useState } from "react";
import { META } from "../config.js";
import { PRIORITY, pct } from "../utils.js";

const LEVEL = {
  urgent: { text: "Urgent", cls: "border-red-500/70 bg-red-500/15 text-red-100", icon: "▲" },
  attention: { text: "Needs attention", cls: "border-amber-400/60 bg-amber-400/10 text-amber-100", icon: "●" },
  routine: { text: "Routine", cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-100", icon: "✓" },
};

/** Caregiver actions from the adaptive model, with Done / Not needed feedback. */
export default function CareActions({ poll, onFeedback, readOnly }) {
  const p = poll.data;
  const [busy, setBusy] = useState(null);
  const [done, setDone] = useState({});

  const act = async (a, action) => {
    setBusy(a.id + action);
    try {
      await onFeedback(a.label, action, a.id);
      setDone((d) => ({ ...d, [a.id]: action }));
    } finally {
      setBusy(null);
    }
  };

  const level = LEVEL[p?.alert?.level] ?? null;
  return (
    <section aria-label="Care actions" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Care actions</h2>
        {level && (
          <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${level.cls}`}>
            <span aria-hidden="true">{level.icon}</span> {level.text}
          </span>
        )}
      </div>

      {!p && <p className="text-slate-400">{poll.error ? `Unavailable (${poll.error.message}).` : "Loading..."}</p>}
      {p && !p.adaptive && (
        <p className="text-sm text-slate-400">This backend has no adaptive model yet (no /insights endpoint).</p>
      )}
      {p?.adaptive && p.actions.length === 0 && <p className="text-slate-400">Nothing needs doing right now.</p>}

      <ul className="space-y-2">
        {p?.actions.map((a) => {
          const pr = PRIORITY[a.priority] ?? PRIORITY.low;
          const m = a.label ? META[a.label] : null;
          const answered = done[a.id];
          return (
            <li key={a.id} className={`rounded-xl border px-3 py-2.5 ${pr.cls}`}>
              <div className="flex items-start gap-3">
                <span className="mt-0.5 text-2xl leading-none" aria-hidden="true">
                  {m?.icon ?? (a.kind === "device" ? "📷" : "👁️")}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-slate-100">{a.title}</span>
                    <span className="rounded border border-current/30 px-1.5 text-[10px] font-bold uppercase tracking-wide">
                      <span aria-hidden="true">{pr.icon}</span> {pr.text}
                    </span>
                    {a.kind === "need" && <span className="text-xs tabular-nums text-slate-300">{pct(a.prob)}</span>}
                  </div>
                  <p className="text-sm text-slate-300">{a.detail}</p>
                  {a.reasons.length > 0 && (
                    <ul className="mt-1 text-xs text-slate-400">
                      {a.reasons.slice(0, 3).map((r) => (
                        <li key={r}>› {r}</li>
                      ))}
                    </ul>
                  )}
                </div>
                {a.label && !readOnly && (
                  <div className="flex shrink-0 flex-col gap-1">
                    {answered ? (
                      <span className="text-xs text-emerald-300">{answered === "done" ? "Marked done" : "Dismissed"}</span>
                    ) : (
                      <>
                        <button
                          disabled={!!busy}
                          onClick={() => act(a, "done")}
                          className="rounded-lg border border-emerald-500/60 px-2 py-0.5 text-xs text-emerald-200 hover:bg-emerald-500/10 disabled:opacity-50"
                        >
                          Done
                        </button>
                        <button
                          disabled={!!busy}
                          onClick={() => act(a, "dismiss")}
                          className="rounded-lg border border-slate-600 px-2 py-0.5 text-xs text-slate-300 hover:bg-slate-700/40 disabled:opacity-50"
                        >
                          Not needed
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      {p?.adaptive && (
        <p className="mt-3 text-xs text-slate-500">
          "Done" and "Not needed" lower that need for 30-45 min and are saved for review.
        </p>
      )}
    </section>
  );
}
