import { GAZE_WINDOW_S, GRID_COLS, GRID_ROWS, ZONES } from "../config.js";

// 6-digit hex + alpha (0-1) -> 8-digit hex
const withAlpha = (hex, a) => hex + Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");

export default function GazeHeatmap({ poll, now }) {
  const events = (poll.data ?? []).filter((e) => e.t >= now - GAZE_WINDOW_S * 1000);

  const counts = Object.fromEntries(ZONES.map((z) => [z.label, 0]));
  for (const e of events) if (e.zone && e.zone in counts) counts[e.zone] += 1;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const maxShare = total ? Math.max(...Object.values(counts)) / total : 0;

  // Zone of the newest sample, if it is fresh: marks "looking now".
  let lookingNow = null;
  if (events.length && poll.updatedAt) {
    const last = events.reduce((a, b) => (b.t > a.t ? b : a));
    if (poll.updatedAt - last.t < 4000) lookingNow = last.zone;
  }

  return (
    <section aria-label="Gaze heatmap" className="h-full rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Gaze heatmap</h2>
        <span className="text-xs text-slate-500">
          last {GAZE_WINDOW_S / 60} min · {total} samples
        </span>
      </div>

      <div className="relative">
        <div
          className="grid gap-2"
          style={{ gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}
        >
          {ZONES.map((z) => {
            const share = total ? counts[z.label] / total : 0;
            const strength = maxShare ? share / maxShare : 0;
            const now_ = lookingNow === z.label;
            return (
              <div
                key={z.label}
                data-zone={z.label}
                className={`flex aspect-[4/3] min-w-0 flex-col items-center justify-center rounded-xl border ${
                  now_ ? "border-white ring-2 ring-white/70" : "border-slate-700"
                }`}
                style={{ background: withAlpha(z.color, 0.07 + 0.7 * strength) }}
              >
                <span className="text-3xl" aria-hidden="true">
                  {z.icon}
                </span>
                <span className="max-w-full truncate text-sm text-slate-200">{z.text}</span>
                <span className="text-xl font-bold tabular-nums">{Math.round(share * 100)}%</span>
              </div>
            );
          })}
        </div>

        {poll.data && total === 0 && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-slate-950/70 text-slate-300">
            No gaze data in the last {GAZE_WINDOW_S / 60} minutes
          </div>
        )}
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Share of gaze samples per tile. A white outline marks where the patient is looking right now.
      </p>
      {poll.error && !poll.data && <p className="mt-2 text-sm text-amber-300">Could not load gaze ({poll.error.message}).</p>}
    </section>
  );
}
