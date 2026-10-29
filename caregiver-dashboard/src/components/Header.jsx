import { ZONES } from "../config.js";

export default function Header({
  patients,
  patientId,
  onPatient,
  mode,
  onMode,
  connError,
  confused,
  onToggleConfused,
  onAddRequest,
}) {
  const live = mode === "live";
  return (
    <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
        <h1 className="text-lg font-bold tracking-tight">Caregiver Dashboard</h1>

        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            live ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-400/20 text-amber-200"
          }`}
        >
          {live ? "LIVE" : "MOCK DATA"}
        </span>
        {live && connError && (
          <span className="rounded-full bg-amber-400/20 px-2.5 py-0.5 text-xs text-amber-200">
            Connection problem, retrying
          </span>
        )}

        <label className="ml-auto flex items-center gap-2 text-sm text-slate-400">
          Patient
          <select
            value={patientId ?? ""}
            onChange={(e) => onPatient(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-100"
          >
            {patients.length === 0 && <option value="">Loading...</option>}
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        {!live && (
          <>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={confused} onChange={onToggleConfused} />
              Simulate confusion
            </label>
            <button
              onClick={onAddRequest}
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm hover:border-sky-400"
            >
              Add request
            </button>
          </>
        )}

        <button
          onClick={() => onMode(live ? "mock" : "live")}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm hover:border-sky-400"
        >
          {live ? "Use mock data" : "Try live backend"}
        </button>
      </div>
    </header>
  );
}

export const LABEL_LIST = ZONES.map((z) => z.label);
