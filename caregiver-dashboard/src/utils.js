// Pure helpers: timestamp parsing, response normalisation, formatting. No React, no DOM.
import { ZONES, GRID_COLS, GRID_ROWS } from "./config.js";

/** Normalised point (0-1 on the pad) -> zone label, or null when outside the pad. */
export function zoneFromXY(x, y) {
  if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) return null;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  const col = Math.min(GRID_COLS - 1, Math.floor(x * GRID_COLS));
  const row = Math.min(GRID_ROWS - 1, Math.floor(y * GRID_ROWS));
  return ZONES[row * GRID_COLS + col].label;
}

/**
 * Timestamp -> epoch ms. FastAPI/SQLAlchemy often return naive ISO strings
 * ("2026-10-17T03:34:35.161" with no Z or offset). The browser would read those as
 * LOCAL time and shift every event by the viewer's UTC offset, so naive = UTC here.
 */
export function parseTs(ts) {
  if (typeof ts === "number") return ts;
  if (typeof ts !== "string") return NaN;
  const naive = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(ts);
  return Date.parse(naive ? ts.replace(" ", "T") + "Z" : ts);
}

/** Accept either a bare array or an object wrapping it ({items: []}, {requests: []}, ...). */
export function asArray(r, ...keys) {
  if (Array.isArray(r)) return r;
  for (const k of keys) if (Array.isArray(r?.[k])) return r[k];
  return [];
}

export function normalizePatients(r) {
  return asArray(r, "patients", "items", "data")
    .map((p) => {
      if (typeof p === "string" || typeof p === "number") return { id: String(p), name: String(p) };
      const id = String(p.id ?? p.patient_id ?? "");
      return { id, name: p.name ?? p.display_name ?? id };
    })
    .filter((p) => p.id);
}

export function normalizeRequests(r) {
  return asArray(r, "requests", "items", "data")
    .map((e) => ({
      patient_id: e.patient_id,
      ts: e.ts,
      t: parseTs(e.ts),
      request: e.request ?? e.label,
      source: e.source ?? "gaze",
    }))
    .filter((e) => Number.isFinite(e.t) && e.request);
}

export function normalizeGaze(r) {
  return asArray(r, "events", "gaze", "items", "data")
    .map((e) => ({
      patient_id: e.patient_id,
      t: parseTs(e.ts),
      x: e.x,
      y: e.y,
      zone: e.zone !== undefined ? e.zone : zoneFromXY(e.x, e.y),
      dwell_ms: e.dwell_ms ?? 0,
      confidence: e.confidence ?? null,
    }))
    .filter((e) => Number.isFinite(e.t));
}

/** Top-3 suggestions sorted by probability, plus the confusion flag and score. */
export function normalizePrediction(r) {
  const suggestions = asArray(r?.suggestions ? r.suggestions : r, "suggestions")
    .map((s) => ({ label: s.label ?? s.request, prob: Number(s.prob ?? s.probability ?? 0) }))
    .filter((s) => s.label)
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
  return {
    suggestions,
    confused: Boolean(r?.confused),
    confusion_score: Number(r?.confusion_score ?? 0),
  };
}

export function timeAgo(t, now) {
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export const pct = (p) => `${Math.round(p * 100)}%`;

export const clock = (t) =>
  new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ---------------------------------------------------------------- adaptive insights
const num = (v, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d);

/** GET /insights -> a normalised object. Also accepts a bare /predict response (older backend). */
export function normalizeInsights(r) {
  const base = normalizePrediction(r);
  const needs = asArray(r?.needs, "needs")
    .map((n) => ({ label: n.label, prob: num(n.prob), reasons: Array.isArray(n.reasons) ? n.reasons : [] }))
    .filter((n) => n.label)
    .sort((a, b) => b.prob - a.prob);
  const ws = r?.webcam && typeof r.webcam === "object" ? r.webcam : { available: false };
  return {
    ...base,
    adaptive: Array.isArray(r?.needs),
    needs: needs.length ? needs : base.suggestions.map((s) => ({ ...s, reasons: [] })),
    alert: r?.alert ?? null,
    actions: asArray(r?.actions, "actions").map((a) => ({
      ...a,
      prob: num(a.prob),
      reasons: Array.isArray(a.reasons) ? a.reasons : [],
    })),
    upcoming: asArray(r?.upcoming, "upcoming"),
    webcam: ws,
    experts: asArray(r?.experts, "experts"),
    model: r?.model ?? null,
    lastRequest: r?.last_request ? { ...r.last_request, t: parseTs(r.last_request.t) } : null,
    feedbackApplied: asArray(r?.feedback_applied, "feedback_applied"),
  };
}

/** GET /windows -> [{t, ...B1 fields}] ascending. */
export function normalizeWindows(r) {
  return asArray(r, "windows", "items", "data")
    .map((w) => ({ t: parseTs(w.start ?? w.ts), simulated: !!w.simulated, ...(w.message ?? w) }))
    .filter((w) => Number.isFinite(w.t))
    .sort((a, b) => a.t - b.t);
}

export function normalizeStats(r) {
  if (!r || typeof r !== "object") return null;
  return {
    n: num(r.n),
    overall: r.overall ?? null,
    last20: r.last_20 ?? null,
    last50: r.last_50 ?? null,
    webcam: r.webcam ?? null,
    noWebcam: r.no_webcam ?? null,
    rolling: asArray(r.rolling).map((p) => ({ i: num(p.i), t: parseTs(p.ts), top1: p.top1, top3: p.top3 })),
    byDay: asArray(r.by_day),
  };
}

export const PRIORITY = {
  high: { text: "High", icon: "▲", cls: "bg-red-500/20 text-red-200 border-red-500/60" },
  medium: { text: "Medium", icon: "●", cls: "bg-amber-400/15 text-amber-100 border-amber-400/50" },
  low: { text: "Low", icon: "○", cls: "bg-slate-700/40 text-slate-300 border-slate-600" },
};
