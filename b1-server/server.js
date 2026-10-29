// Node.js server for the Team B (B1) integration.
//   GazePad  --POST /window-->  this server  <--GET /latest (every 3 s)--  Team B's B1 script
// No API key. B1 only ever sees GET /latest: a JSON list with the latest completed 10 s window per patient.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { validate } from "./schema.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8100);
const DATA_DIR = process.env.B1_DATA_DIR || path.join(HERE, "data");
const ALLOW_SIMULATED = process.env.B1_ALLOW_SIMULATED === "1"; // mock-mode (mouse) windows are never exposed unless set
const CORS_ORIGINS = (process.env.B1_CORS_ORIGINS || "http://localhost:5173,http://127.0.0.1:5173").split(",");
const STALE_S = Number(process.env.B1_STALE_S || 0); // 0 = keep serving the last window forever (B1 de-duplicates)

fs.mkdirSync(DATA_DIR, { recursive: true });
const LATEST_FILE = path.join(DATA_DIR, "latest.json");
const LOG_FILE = path.join(DATA_DIR, "events.jsonl");

// user_id -> { message, start (epoch ms of window start), received_at (ISO) }
let latest = {};
try {
  latest = JSON.parse(fs.readFileSync(LATEST_FILE, "utf8"));
} catch {
  /* first run */
}
const stats = { received: 0, rejected: 0, held_simulated: 0, stale_ignored: 0, polls: 0, last_poll_at: null };

function log(type, extra = {}) {
  fs.appendFile(LOG_FILE, JSON.stringify({ at: new Date().toISOString(), type, ...extra }) + "\n", () => {});
}

function send(res, status, body, req) {
  const origin = req?.headers.origin;
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (origin && (CORS_ORIGINS.includes("*") || CORS_ORIGINS.includes(origin))) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Access-Control-Allow-Headers"] = "Content-Type";
    headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS";
  }
  res.writeHead(status, headers);
  res.end(JSON.stringify(body));
}

const readBody = (req) =>
  new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (c) => {
      data += c;
      if (data.length > 100_000) reject(new Error("body too large"));
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });

function visibleLatest() {
  const cutoff = STALE_S ? Date.now() - STALE_S * 1000 : 0;
  return Object.values(latest)
    .filter((e) => e.start >= cutoff)
    .sort((a, b) => (a.message.User_ID < b.message.User_ID ? -1 : 1))
    .map((e) => e.message);
}

export const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  try {
    if (req.method === "OPTIONS") return send(res, 204, {}, req);

    // ---- what Team B calls ----
    if (req.method === "GET" && url.pathname === "/latest") {
      stats.polls += 1;
      stats.last_poll_at = new Date().toISOString();
      return send(res, 200, visibleLatest(), req); // [] when nothing yet: B1 simply ignores it
    }
    if (req.method === "GET" && url.pathname.startsWith("/latest/")) {
      const e = latest[decodeURIComponent(url.pathname.slice(8))];
      return e ? send(res, 200, e.message, req) : send(res, 404, { error: "no window for that user yet" }, req);
    }

    // ---- what GazePad calls ----
    if (req.method === "POST" && url.pathname === "/window") {
      let env;
      try {
        env = JSON.parse(await readBody(req));
      } catch {
        return send(res, 400, { detail: ["body must be JSON"] }, req);
      }
      const { message, key, simulated } = env || {};
      const errors = validate(message);
      if (errors.length) {
        stats.rejected += 1;
        log("rejected", { errors, key });
        return send(res, 422, { detail: errors }, req);
      }
      stats.received += 1;
      if (simulated && !ALLOW_SIMULATED) {
        stats.held_simulated += 1;
        log("held_simulated", { key, user: message.User_ID, window: message.Time_Window });
        return send(res, 200, { status: "held_simulated" }, req);
      }
      // window start (epoch ms) comes from the key "patient|startMs"; used so a late, older window never replaces a newer one
      const start = Number(String(key || "").split("|")[1]) || Date.now();
      const cur = latest[message.User_ID];
      if (cur && cur.start >= start) {
        stats.stale_ignored += 1;
        log("stale_ignored", { key, user: message.User_ID, window: message.Time_Window });
        return send(res, 200, { status: "stale_ignored" }, req);
      }
      latest[message.User_ID] = { message, start, received_at: new Date().toISOString() };
      fs.writeFile(LATEST_FILE, JSON.stringify(latest), () => {});
      log("accepted", { key, message });
      return send(res, 200, { status: "accepted" }, req);
    }

    // ---- for us ----
    if (req.method === "GET" && url.pathname === "/status")
      return send(res, 200, { ...stats, allow_simulated: ALLOW_SIMULATED, patients: Object.keys(latest) }, req);
    if (req.method === "GET" && url.pathname === "/logs") {
      let lines = [];
      try {
        lines = fs.readFileSync(LOG_FILE, "utf8").trim().split("\n").filter(Boolean).slice(-Number(url.searchParams.get("n") || 50));
      } catch {
        /* no log yet */
      }
      return send(res, 200, lines.map((l) => JSON.parse(l)), req);
    }
    if (req.method === "GET" && url.pathname === "/health") return send(res, 200, { ok: true }, req);
    send(res, 404, { error: "not found" }, req);
  } catch (e) {
    send(res, 500, { error: String(e.message || e) }, req);
  }
});

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  server.listen(PORT, () => {
    console.log(`b1-server on :${PORT}  (B1 polls GET /latest; GazePad posts /window; simulated windows ${ALLOW_SIMULATED ? "EXPOSED" : "held"})`);
  });
}
