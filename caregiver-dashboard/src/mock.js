// Mock backend: same methods as liveApi, no network. Deterministic (seeded) so charts do not
// jitter between polls. Includes demo controls to simulate confusion and new requests.
import { ZONES, GRID_COLS, GRID_ROWS } from "./config.js";
import {
  normalizePatients,
  normalizeRequests,
  normalizeGaze,
  normalizePrediction,
  normalizeInsights,
  normalizeWindows,
  normalizeStats,
} from "./utils.js";

const LABELS = ZONES.map((z) => z.label);
const DAY = 86400000;

// ---- seeded randomness ---------------------------------------------------------
function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = (key) => mulberry32(hashStr(key));

// ---- patients and their daily routines: [label, hour, minute] -----------------
const PATIENTS = [
  { id: "p1", name: "Patient A (mock)" },
  { id: "p2", name: "Patient B (mock)" },
  { id: "p3", name: "Patient C (mock)" },
];
const ROUTINES = {
  p1: [["water", 10, 0], ["bathroom", 11, 30], ["food", 13, 0], ["rest", 15, 0], ["water", 16, 30], ["food", 19, 0], ["rest", 21, 30]],
  p2: [["food", 8, 30], ["water", 11, 0], ["bathroom", 12, 30], ["rest", 14, 0], ["water", 17, 0], ["pain", 18, 30], ["food", 19, 30]],
  p3: [["rest", 9, 0], ["water", 10, 30], ["food", 12, 30], ["help", 15, 0], ["bathroom", 16, 0], ["food", 18, 30], ["rest", 20, 0]],
};
const routineFor = (id) => ROUTINES[id] || ROUTINES.p1;

// ---- demo state ----------------------------------------------------------------
const confusedSince = new Map(); // patientId -> epoch ms
const extraRequests = []; // requests added with the "Add request" button
const feedback = []; // {patient_id, t, label, action}

// ---- request history -----------------------------------------------------------
function dayStart(ms) {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function historyFor(patientId, toMs) {
  const out = [];
  const today = dayStart(toMs);
  for (let d = -2; d <= 0; d++) {
    const start = today + d * DAY;
    const r = rng(`${patientId}|${start}`);
    for (const [label, h, m] of routineFor(patientId)) {
      const jitter = (r() * 2 - 1) * 20 * 60000; // +-20 min
      const noisy = r() < 0.1; // 10% of requests are not the usual one
      out.push({
        t: start + (h * 60 + m) * 60000 + jitter,
        label: noisy ? LABELS[Math.floor(r() * LABELS.length)] : label,
      });
    }
    for (let i = 0; i < 2; i++) {
      if (r() < 0.5) out.push({ t: start + r() * DAY, label: LABELS[Math.floor(r() * LABELS.length)] });
    }
  }
  return out;
}

// ---- time-of-day prior (what the real model falls back to when confused) -----
const minutesOfDay = (ms) => {
  const d = new Date(ms);
  return d.getHours() * 60 + d.getMinutes();
};

function prior(patientId, ms) {
  const now = minutesOfDay(ms);
  const w = Object.fromEntries(LABELS.map((l) => [l, 0.02]));
  for (const [label, h, m] of routineFor(patientId)) {
    let dist = Math.abs(now - (h * 60 + m));
    dist = Math.min(dist, 1440 - dist);
    w[label] += Math.exp(-(dist * dist) / (2 * 40 * 40));
  }
  const sum = Object.values(w).reduce((a, b) => a + b, 0);
  return LABELS.map((l) => ({ label: l, prob: w[l] / sum })).sort((a, b) => b.prob - a.prob);
}

// ---- gaze samples (10 Hz) --------------------------------------------------------
function gazeSamples(patientId, seconds, nowMs) {
  const STEP = 100;
  const start = Math.floor((nowMs - seconds * 1000) / STEP) * STEP;
  const cs = confusedSince.get(patientId);
  const likely = prior(patientId, nowMs)[0].label;
  const out = [];
  for (let t = start; t <= nowMs; t += STEP) {
    const confused = cs != null && t >= cs;
    const seg = confused ? 400 : 4000; // confused gaze hops around ten times faster
    const bucket = Math.floor(t / seg);
    const rb = rng(`${patientId}|${confused ? "c" : "n"}|${bucket}`);
    const label =
      !confused && rb() < 0.55 ? likely : LABELS[Math.floor(rb() * LABELS.length)];
    const idx = LABELS.indexOf(label);
    const col = idx % GRID_COLS;
    const row = Math.floor(idx / GRID_COLS);
    const rs = rng(`${patientId}|${t}`);
    out.push({
      patient_id: patientId,
      ts: new Date(t).toISOString(),
      x: +((col + 0.12 + 0.76 * rs()) / GRID_COLS).toFixed(4),
      y: +((row + 0.12 + 0.76 * rs()) / GRID_ROWS).toFixed(4),
      zone: label,
      dwell_ms: Math.max(0, t - bucket * seg),
      confidence: 0.9,
    });
  }
  return out;
}

// ---- webcam windows (one per 10 s) ----------------------------------------------------
function windowAt(patientId, t) {
  const r = rng(`${patientId}|w|${t}`);
  const cs = confusedSince.get(patientId);
  const c = cs != null && t >= cs ? Math.min(1, (t - cs) / 60000) : 0;
  const dev = 4 + 3.2 * c + r() * 1.5;
  const fix = Math.max(0.1, 0.32 - 0.15 * c + (r() - 0.5) * 0.08);
  return {
    t,
    Mean_Gaze_Deviation: +dev.toFixed(3),
    Max_Gaze_Deviation: +(dev + 3 + r() * 2).toFixed(3),
    Blink_Rate: Math.floor(r() * 5),
    Avg_Fixation_Duration: +fix.toFixed(3),
    Max_Fixation_Duration: +(fix * 2.5).toFixed(3),
    Head_Movement: +((2 + r() * 3) * (r() < 0.04 ? 6 : 1)).toFixed(3),
    Gaze_Stability: +(0.36 - 0.13 * c + (r() - 0.5) * 0.04).toFixed(3),
    Emergency_Button: 0,
  };
}

function mockInsights(patientId, ms) {
  const confused = confusedSince.has(patientId);
  let needs = prior(patientId, ms);
  for (const f of feedback) {
    if (f.patient_id === patientId && ms - f.t < 30 * 60000) {
      needs = needs.map((n) => (n.label === f.label ? { ...n, prob: n.prob * 0.4 } : n));
    }
  }
  const sum = needs.reduce((a, n) => a + n.prob, 0);
  needs = needs.map((n) => ({ ...n, prob: n.prob / sum })).sort((a, b) => b.prob - a.prob);
  const top = needs[0];
  const meta = ZONES.find((z) => z.label === top.label);
  needs = needs.map((n, i) => ({
    ...n,
    reasons: i === 0 ? [`Usually asks for ${ZONES.find((z) => z.label === n.label).text} around this time (mock routine)`] : [],
  }));
  const states = { distress: 0.12, fatigue: 0.15, confusion: confused ? 0.82 : 0.1 };
  const actions = [];
  if (confused) {
    actions.push({ id: "state:confusion", kind: "state", priority: "medium", title: "Reassure and reorient",
      detail: "Speak calmly, explain where they are, and offer the GazePad choices one at a time.", prob: 0.82,
      reasons: ["gaze deviation higher than usual, gaze stability lower than usual"] });
  }
  if (top.prob >= 0.25) {
    actions.push({ id: `need:${top.label}`, kind: "need", label: top.label, priority: top.prob >= 0.45 ? "medium" : "low",
      title: `Offer ${meta.text.toLowerCase()}`, detail: "Mock suggestion from the daily routine.", prob: top.prob, reasons: needs[0].reasons });
  }
  const level = actions.some((a) => a.priority === "medium") ? "attention" : "routine";
  return {
    needs,
    suggestions: needs.slice(0, 3).map(({ label, prob }) => ({ label, prob })),
    confused,
    confusion_score: confused ? 0.84 : 0.14,
    alert: { level, text: actions[0]?.title ?? "No pressing needs predicted" },
    actions,
    upcoming: [],
    webcam: {
      available: true, personal_baseline: true, n_recent: 9, states,
      z: { head: 0.2, stability: confused ? -2.6 : 0.1, deviation: confused ? 2.4 : 0.1, blink: 0.1, fixation: confused ? -1.4 : 0 },
      now: { head: 3.1, stability: confused ? 0.24 : 0.36, deviation: confused ? 7.1 : 4.6, blink: 2, fixation: confused ? 0.18 : 0.32 },
      baseline: { head: 3.0, stability: 0.36, deviation: 4.6, blink: 2, fixation: 0.32 },
    },
    experts: [
      { name: "routine", text: "Daily routine", weight: 0.46, awake: true, top: top.label },
      { name: "learned", text: "Personal learned model", weight: 0.31, awake: true, top: top.label },
      { name: "webcam", text: "Webcam state", weight: 0.23, awake: confused, top: confused ? "help" : null },
    ],
    model: { n_learned: 42, rolling_top1: 0.71, rolling_top3: 0.93 },
  };
}

// ---- the API -------------------------------------------------------------------------
export const mockApi = {
  kind: "mock",

  async getPatients() {
    return normalizePatients(PATIENTS);
  },

  async getRequests(patientId, from, to) {
    const f = from.getTime();
    const t = to.getTime();
    const rows = [
      ...historyFor(patientId, t).map((e) => ({
        patient_id: patientId,
        ts: new Date(e.t).toISOString(),
        request: e.label,
        source: "simulated",
      })),
      ...extraRequests.filter((e) => e.patient_id === patientId),
    ].filter((e) => {
      const ms = Date.parse(e.ts);
      return ms >= f && ms <= t;
    });
    return normalizeRequests(rows);
  },

  async getGaze(patientId, seconds) {
    return normalizeGaze(gazeSamples(patientId, seconds, Date.now()));
  },

  async predict(patientId, ts) {
    const confused = confusedSince.has(patientId);
    const ms = Date.parse(ts) || Date.now();
    return normalizePrediction({
      suggestions: prior(patientId, ms).slice(0, 3),
      confused,
      confusion_score: confused ? 0.84 : 0.14,
    });
  },

  async getInsights(patientId) {
    return normalizeInsights(mockInsights(patientId, Date.now()));
  },

  async getWindows(patientId, minutes) {
    const now = Math.floor(Date.now() / 10000) * 10000;
    const out = [];
    for (let t = now - minutes * 60000; t < now; t += 10000) {
      out.push({ start: new Date(t).toISOString(), message: windowAt(patientId, t), simulated: true });
    }
    return normalizeWindows(out);
  },

  async getStats(patientId) {
    const r = rng(`${patientId}|stats`);
    const rolling = [];
    let a1 = 0.35;
    for (let i = 1; i <= 60; i++) {
      a1 = Math.min(0.85, a1 + 0.012 + (r() - 0.5) * 0.04);
      rolling.push({ i, ts: new Date(Date.now() - (60 - i) * 3600000).toISOString(), top1: +a1.toFixed(2), top3: +Math.min(1, a1 + 0.2).toFixed(2) });
    }
    return normalizeStats({ n: 60, overall: { n: 60, top1: 0.66, top3: 0.88 }, last_20: { n: 20, top1: 0.78, top3: 0.95 }, rolling });
  },

  async sendFeedback(patientId, label, action) {
    feedback.push({ patient_id: patientId, t: Date.now(), label, action });
    return { ok: true };
  },

  // ---- demo controls (mock mode only) ----
  isConfused: (patientId) => confusedSince.has(patientId),
  setConfused(patientId, on) {
    if (on) confusedSince.set(patientId, Date.now());
    else confusedSince.delete(patientId);
  },
  addRequest(patientId, label) {
    extraRequests.push({
      patient_id: patientId,
      ts: new Date().toISOString(),
      request: label,
      source: "simulated",
    });
  },
};
