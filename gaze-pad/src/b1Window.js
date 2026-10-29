// Turns 10 seconds of gaze samples into the 10 numbers Team B's B1 model expects
// (B1_Data_Requirements.pdf). Pure functions, no React or DOM, so they are unit tested.
import { headStepDeg } from "./faceMetrics.js";

export const WINDOW_MS = 10000;
export const SLOT_MS = 100; // B1 wants exactly 10 samples per second
export const SLOTS = WINDOW_MS / SLOT_MS;

const RAD = 180 / Math.PI;
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const round = (v, d = 3) => Math.round(v * 10 ** d) / 10 ** d;
const p2 = (n) => String(n).padStart(2, "0");

/** Our patient id -> B1 User_ID: 1, "1" or "p1" -> "USER001". */
export function userIdFor(patientId) {
  const s = String(patientId ?? "").trim();
  const digits = s.match(/\d+/);
  if (digits) return `USER${digits[0].padStart(3, "0")}`;
  return `USER${s.toUpperCase().replace(/[^A-Z0-9]/g, "") || "000"}`;
}

const clock = (ms, tz) => {
  const d = new Date(ms);
  return tz === "utc"
    ? `${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}:${p2(d.getUTCSeconds())}`
    : `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
};

/** "HH:MM:SS-HH:MM:SS" for the window starting at startMs (local clock time unless tz is "utc"). */
export function formatTimeWindow(startMs, tz = "local") {
  return `${clock(startMs, tz)}-${clock(startMs + WINDOW_MS, tz)}`;
}

/**
 * Resample irregular samples onto exactly 100 slots of 100 ms.
 * Each slot takes the newest sample at or before the end of the slot, if it is at most
 * maxStaleMs old; otherwise the slot is null (no gaze data, e.g. face lost).
 * samples: [{t, x, y, rot}] sorted by t ascending (t in epoch ms).
 */
export function buildSlots(samples, startMs, maxStaleMs = 250) {
  const slots = new Array(SLOTS).fill(null);
  let k = 0;
  let last = null;
  for (let i = 0; i < SLOTS; i++) {
    const slotEnd = startMs + (i + 1) * SLOT_MS;
    while (k < samples.length && samples[k].t <= slotEnd) last = samples[k++];
    if (last && slotEnd - last.t <= maxStaleMs) slots[i] = last;
  }
  return slots;
}

/**
 * The 8 numeric B1 features for one window.
 *
 * slots   output of buildSlots (x, y normalised 0-1 on the pad; rot = head rotation or null)
 * o       { padW, padH   pad size in CSS px,
 *           cmPerPx      physical size of one CSS px,
 *           viewDistCm   eye-to-screen distance,
 *           fixationDeg  gaze must stay within this many degrees to count as one fixation (2),
 *           blinks, emergencies   counts for the window }
 *
 * Gaze deviation = angle between the gaze point and one fixed reference point, the screen
 * (pad) centre, the same for every window, as Team B's updated spec requires.
 * Gaze_Stability = 1 / (1 + population std of the deviation).
 */
export function computeWindowFeatures(slots, o) {
  const { padW, padH, cmPerPx, viewDistCm, fixationDeg = 2, blinks = 0, emergencies = 0 } = o;
  const toDeg = (cm) => Math.atan(cm / viewDistCm) * RAD;
  const cmX = (x) => clamp01(x) * padW * cmPerPx;
  const cmY = (y) => clamp01(y) * padH * cmPerPx;

  // deviation = angle between the gaze point and ONE fixed reference point, the pad (screen) centre
  const dev = [];
  for (const s of slots) {
    if (!s) continue;
    dev.push(toDeg(Math.hypot((clamp01(s.x) - 0.5) * padW * cmPerPx, (clamp01(s.y) - 0.5) * padH * cmPerPx)));
  }
  const n = dev.length;
  const mean = n ? dev.reduce((a, b) => a + b, 0) / n : 0;
  const max = n ? Math.max(...dev) : 0;
  const std = n ? Math.sqrt(dev.reduce((a, b) => a + (b - mean) ** 2, 0) / n) : 0;

  // fixations: runs of consecutive samples that stay within fixationDeg of the run's centre
  const durations = [];
  let i = 0;
  while (i < slots.length) {
    if (!slots[i]) {
      i += 1;
      continue;
    }
    let sumX = cmX(slots[i].x);
    let sumY = cmY(slots[i].y);
    let count = 1;
    let j = i + 1;
    while (j < slots.length && slots[j]) {
      const px = cmX(slots[j].x);
      const py = cmY(slots[j].y);
      if (toDeg(Math.hypot(px - sumX / count, py - sumY / count)) > fixationDeg) break;
      sumX += px;
      sumY += py;
      count += 1;
      j += 1;
    }
    durations.push((count * SLOT_MS) / 1000);
    i = j;
  }
  const avgFix = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;
  const maxFix = durations.length ? Math.max(...durations) : 0;

  // head movement: largest sqrt(dyaw^2 + dpitch^2 + droll^2) between consecutive samples
  let head = 0;
  for (let k = 1; k < slots.length; k++) {
    const a = slots[k - 1];
    const b = slots[k];
    if (a && b && a.rot && b.rot) head = Math.max(head, headStepDeg(a.rot, b.rot));
  }

  return {
    Mean_Gaze_Deviation: round(mean),
    Max_Gaze_Deviation: round(max),
    Blink_Rate: blinks,
    Avg_Fixation_Duration: round(avgFix),
    Max_Fixation_Duration: round(maxFix),
    Head_Movement: round(head),
    Gaze_Stability: round(1 / (1 + std)),
    Emergency_Button: emergencies,
  };
}
