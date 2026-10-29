// Strict validator for the Group B1 input format. Exact field names, JSON numbers only.
export const FIELDS = {
  User_ID: "text",
  Time_Window: "text",
  Mean_Gaze_Deviation: "number", // degrees
  Max_Gaze_Deviation: "number", // degrees
  Blink_Rate: "count", // blinks in the 10 s window
  Avg_Fixation_Duration: "number", // seconds
  Max_Fixation_Duration: "number", // seconds
  Head_Movement: "number", // degrees per sample
  Gaze_Stability: "number", // 0..1
  Emergency_Button: "count",
};

export const EXAMPLE = {
  User_ID: "USER001", Time_Window: "14:35:30-14:35:40",
  Mean_Gaze_Deviation: 4.1, Max_Gaze_Deviation: 8.7, Blink_Rate: 3,
  Avg_Fixation_Duration: 0.34, Max_Fixation_Duration: 0.8, Head_Movement: 1.2,
  Gaze_Stability: 0.42, Emergency_Button: 0,
};

const RE = /^(\d{2}):(\d{2}):(\d{2})-(\d{2}):(\d{2}):(\d{2})$/;
const secs = (h, m, s) => +h * 3600 + +m * 60 + +s;

/** Returns a list of problems; empty means valid. */
export function validate(msg) {
  if (msg === null || typeof msg !== "object" || Array.isArray(msg)) return ["message must be a JSON object"];
  const errors = [];
  const lower = Object.fromEntries(Object.keys(FIELDS).map((k) => [k.toLowerCase(), k]));
  for (const k of Object.keys(msg)) {
    if (!(k in FIELDS)) {
      const hint = lower[k.toLowerCase()] ? ` (did you mean '${lower[k.toLowerCase()]}'?)` : "";
      errors.push(`unexpected field '${k}'${hint}: do not rename or add fields`);
    }
  }
  for (const k of Object.keys(FIELDS)) if (!(k in msg)) errors.push(`missing field '${k}'`);
  if (errors.length) return errors;

  for (const [k, kind] of Object.entries(FIELDS)) {
    const v = msg[k];
    if (kind === "text") {
      if (typeof v !== "string" || !v.trim()) errors.push(`'${k}' must be non-empty text`);
    } else if (typeof v !== "number") errors.push(`'${k}' must be a JSON number`);
    else if (!Number.isFinite(v)) errors.push(`'${k}' must be a finite number`);
    else if (v < 0) errors.push(`'${k}' must not be negative (got ${v})`);
    else if (kind === "count" && !Number.isInteger(v)) errors.push(`'${k}' must be a whole number (got ${v})`);
  }
  if (errors.length) return errors;

  const m = RE.exec(msg.Time_Window);
  if (!m) errors.push("'Time_Window' must look like HH:MM:SS-HH:MM:SS");
  else {
    const n = m.slice(1).map(Number);
    if (n[0] > 23 || n[3] > 23 || [n[1], n[2], n[4], n[5]].some((x) => x > 59)) errors.push("'Time_Window' has an invalid clock time");
    else if ((((secs(m[4], m[5], m[6]) - secs(m[1], m[2], m[3])) % 86400) + 86400) % 86400 !== 10)
      errors.push("'Time_Window' must span exactly 10 seconds");
  }
  const eps = 1e-9;
  if (msg.Gaze_Stability > 1 + eps) errors.push("'Gaze_Stability' must be between 0 and 1");
  if (msg.Max_Gaze_Deviation + eps < msg.Mean_Gaze_Deviation) errors.push("'Max_Gaze_Deviation' cannot be smaller than 'Mean_Gaze_Deviation'");
  if (msg.Max_Fixation_Duration + eps < msg.Avg_Fixation_Duration) errors.push("'Max_Fixation_Duration' cannot be smaller than 'Avg_Fixation_Duration'");
  if (msg.Max_Fixation_Duration > 10 + eps) errors.push("'Max_Fixation_Duration' cannot exceed the 10 s window");
  return errors;
}
