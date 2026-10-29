// Shared contract constants. Keep in sync with the other three modules.

// import.meta.env is undefined outside Vite (e.g. node --test), hence the "?."
export const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";
export const DEFAULT_PATIENT_ID = "p1";

// Fixed request labels. Grid order is row-major: 3 columns x 2 rows.
export const GRID_COLS = 3;
export const GRID_ROWS = 2;

export const ZONES = [
  { label: "water", text: "Water", icon: "\u{1F4A7}", color: "#38bdf8" },
  { label: "food", text: "Food", icon: "\u{1F37D}️", color: "#fbbf24" },
  { label: "help", text: "Help", icon: "\u{1F198}", color: "#f87171" },
  { label: "pain", text: "Pain", icon: "\u{1F915}", color: "#c084fc" },
  { label: "bathroom", text: "Bathroom", icon: "\u{1F6BB}", color: "#2dd4bf" },
  { label: "rest", text: "Rest", icon: "\u{1F6CF}️", color: "#818cf8" },
];

export const LABELS = ZONES.map((z) => z.label);

// Timing
export const DWELL_MS = 1500; // dwell needed to fire a request
export const DEBOUNCE_MS = 250; // gaze must stay in a new zone this long before it counts
export const TICK_MS = 50; // processing loop (20 Hz)
export const SAMPLE_MS = 100; // one gaze event every 100 ms (10 Hz)
export const FLUSH_MS = 1000; // batch POST /gaze every second
export const STALE_MS = 600; // no tracker output for this long = no gaze

// Team B (B1) integration. Defaults can be overridden with URL params, see App.jsx.
export const B1 = {
  relayBase: import.meta.env?.VITE_B1_RELAY || "http://localhost:8100",
  screenWidthCm: 30, // physical width of the screen (about a 13-14 inch laptop)
  viewDistCm: 50, // eye-to-screen distance
  tz: "local", // Time_Window clock: "local" or "utc"
  emergencyLabel: "help", // selecting this tile counts as Emergency_Button
  minValid: 50, // skip windows with fewer than this many of 100 gaze slots
  fixationDeg: 2, // gaze within this angle counts as one fixation
};

// Calibration
export const CAL_SETTLE_MS = 800; // time for the eyes to land on a point
export const CAL_RECORD_MS = 1400; // time spent recording samples per point
export const CAL_SAMPLE_EVERY_MS = 80;
export const CAL_POINTS = [
  [0.1, 0.12], [0.5, 0.12], [0.9, 0.12],
  [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
  [0.1, 0.88], [0.5, 0.88], [0.9, 0.88],
];
