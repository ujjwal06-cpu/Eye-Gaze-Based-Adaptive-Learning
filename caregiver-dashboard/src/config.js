// Shared contract constants. Keep in sync with the other modules.

// import.meta.env is undefined outside Vite (e.g. node --test), hence the "?."
export const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";

export const POLL = {
  patientsMs: 30000,
  requestsMs: 5000,
  gazeMs: 2000,
  predictMs: 10000, // "Suggested now" / care actions refresh
  windowsMs: 10000, // webcam windows arrive every 10 s
  statsMs: 30000, // adaptive model accuracy
};

export const GAZE_WINDOW_S = 300; // heatmap window: last 5 minutes
export const TIMELINE_HOURS = 24;
export const FEED_LIMIT = 15;

// Grid order is row-major 3 cols x 2 rows, identical to GazePad.
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

export const META = Object.fromEntries(ZONES.map((z) => [z.label, z]));
export const URGENT = new Set(["help", "pain"]);
