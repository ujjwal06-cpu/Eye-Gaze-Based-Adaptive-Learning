// Uplink to the FastAPI backend.
//  - gaze events: buffered, batched every FLUSH_MS, lossy if the backend is down for long
//  - request events: queued and persisted in localStorage, retried until delivered
//  - B1 windows: sent to the B1 server exactly as before (same URL, same envelope)
//  - outbox (new): session parameters, session events and a copy of every B1 window for the
//    backend database, queued in localStorage and sent in order. An older backend without
//    these endpoints answers 404 and the items are dropped, so nothing else is affected.
import { FLUSH_MS } from "./config.js";

const PENDING_KEY = "gazepad.pendingRequests";
const WINDOWS_KEY = "gazepad.pendingWindows";
const OUTBOX_KEY = "gazepad.outbox";
const MAX_OUTBOX = 2000;
const MAX_PENDING_WINDOWS = 360; // one hour of 10 s windows
const MAX_GAZE_BUFFER = 5000;
const MAX_BATCH = 500;
const MAX_PENDING_REQUESTS = 200;

async function postJSON(url, body, timeoutMs = 4000, method = "POST") {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
      keepalive: true,
    });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}

// 4xx (except timeout / rate limit) means the payload itself is bad: retrying is pointless.
const isPermanent = (err) =>
  err.status >= 400 && err.status < 500 && err.status !== 408 && err.status !== 429;

function loadPending(key = PENDING_KEY) {
  try {
    return JSON.parse(localStorage.getItem(key) || "[]");
  } catch {
    return [];
  }
}
function savePending(list, key = PENDING_KEY) {
  try {
    localStorage.setItem(key, JSON.stringify(list));
  } catch {
    /* storage full or blocked: queue stays in memory only */
  }
}

export class Uplink {
  constructor({ apiBase, relayBase, onStatus } = {}) {
    this.apiBase = apiBase;
    this.relayBase = relayBase; // B1 relay; falsy disables window sending
    this.windows = relayBase ? loadPending(WINDOWS_KEY) : [];
    this.busyWin = false;
    this.relayOnline = null;
    this.windowsSent = 0;
    this.onStatus = onStatus;
    this.gaze = [];
    this.requests = loadPending();
    this.busyGaze = false;
    this.busyReq = false;
    this.online = null; // null = nothing sent yet
    this.outbox = loadPending(OUTBOX_KEY);
    this.busyOut = false;
    this.timer = null;
    this.onHide = () => {
      if (document.visibilityState === "hidden") this.flush();
    };
  }

  start() {
    this.timer = setInterval(() => this.flush(), FLUSH_MS);
    document.addEventListener("visibilitychange", this.onHide);
    this.flush();
  }

  stop() {
    clearInterval(this.timer);
    document.removeEventListener("visibilitychange", this.onHide);
    this.flush();
  }

  addGaze(ev) {
    this.gaze.push(ev);
    if (this.gaze.length > MAX_GAZE_BUFFER) this.gaze.splice(0, this.gaze.length - MAX_GAZE_BUFFER);
  }

  sendRequest(ev) {
    this.requests.push(ev);
    if (this.requests.length > MAX_PENDING_REQUESTS) this.requests.shift();
    savePending(this.requests);
    this.flushRequests();
  }

  /** Queue a finished 10 s window envelope {message, key, simulated}; persisted and sent in order. */
  sendWindow(env) {
    if (!this.relayBase) return;
    this.windows.push(env);
    if (this.windows.length > MAX_PENDING_WINDOWS) this.windows.shift();
    savePending(this.windows, WINDOWS_KEY);
    this.flushWindows();
  }

  /** Queue a call to the backend (session parameters, events, stored windows). Kept in order. */
  enqueue(method, path, body) {
    this.outbox.push({ method, path, body });
    if (this.outbox.length > MAX_OUTBOX) this.outbox.splice(0, this.outbox.length - MAX_OUTBOX);
    savePending(this.outbox, OUTBOX_KEY);
    this.flushOutbox();
  }

  /** Store a copy of a B1 window in the backend database (the B1 server copy is sendWindow). */
  storeWindow(env, extra = {}) {
    this.enqueue("POST", "/windows", { ...env, ...extra });
  }

  startSession(session) {
    this.enqueue("POST", "/sessions", session);
  }

  sessionEvent(sessionId, type, data = {}) {
    if (!sessionId) return;
    this.enqueue("POST", `/sessions/${encodeURIComponent(sessionId)}/events`, {
      ts: new Date().toISOString(),
      type,
      data,
    });
  }

  /** What the adaptive model expects this patient to need now. Resolves null if unavailable. */
  async insights(patientId) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5000);
    try {
      const res = await fetch(`${this.apiBase}/insights?patient_id=${encodeURIComponent(patientId)}`, {
        signal: ctrl.signal,
      });
      return res.ok ? await res.json() : null;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  status() {
    return {
      online: this.online,
      pendingGaze: this.gaze.length,
      pendingRequests: this.requests.length,
      pendingWindows: this.windows.length,
      pendingOutbox: this.outbox.length,
      relayOnline: this.relayOnline,
      windowsSent: this.windowsSent,
    };
  }

  emit() {
    this.onStatus?.(this.status());
  }

  async flush() {
    await Promise.all([this.flushGaze(), this.flushRequests(), this.flushWindows(), this.flushOutbox()]);
    this.emit();
  }

  async flushOutbox() {
    if (this.busyOut || !this.outbox.length) return;
    this.busyOut = true;
    try {
      while (this.outbox.length) {
        const item = this.outbox[0];
        try {
          await postJSON(`${this.apiBase}${item.path}`, item.body, 4000, item.method);
        } catch (err) {
          if (!isPermanent(err)) break; // backend down: keep the order, retry later
          if (err.status !== 404) console.warn("[GazePad] backend rejected", item.path, err.message);
        }
        this.outbox.shift();
        savePending(this.outbox, OUTBOX_KEY);
      }
    } finally {
      this.busyOut = false;
    }
  }

  async flushGaze() {
    if (this.busyGaze || !this.gaze.length) return;
    this.busyGaze = true;
    const batch = this.gaze.splice(0, MAX_BATCH);
    try {
      await postJSON(`${this.apiBase}/gaze`, batch);
      this.online = true;
    } catch (err) {
      if (isPermanent(err)) {
        console.warn("[GazePad] /gaze rejected, dropping batch:", err.message);
      } else {
        this.online = false;
        this.gaze.unshift(...batch);
        if (this.gaze.length > MAX_GAZE_BUFFER) this.gaze.splice(0, this.gaze.length - MAX_GAZE_BUFFER);
      }
    } finally {
      this.busyGaze = false;
    }
  }

  async flushRequests() {
    if (this.busyReq || !this.requests.length) return;
    this.busyReq = true;
    try {
      // Send oldest first; stop at the first transient failure to keep order.
      while (this.requests.length) {
        const ev = this.requests[0];
        try {
          await postJSON(`${this.apiBase}/requests`, ev);
          this.online = true;
        } catch (err) {
          if (isPermanent(err)) {
            console.warn("[GazePad] /requests rejected, dropping:", err.message, ev);
          } else {
            this.online = false;
            break;
          }
        }
        this.requests.shift();
        savePending(this.requests);
      }
    } finally {
      this.busyReq = false;
      this.emit();
    }
  }

  async flushWindows() {
    if (this.busyWin || !this.relayBase || !this.windows.length) return;
    this.busyWin = true;
    try {
      // oldest first; stop at the first transient failure to keep order
      while (this.windows.length) {
        const env = this.windows[0];
        try {
          await postJSON(`${this.relayBase}/window`, env);
          this.relayOnline = true;
          this.windowsSent += 1;
        } catch (err) {
          if (isPermanent(err)) {
            console.warn("[GazePad] B1 relay rejected a window, dropping:", err.message, env);
          } else {
            this.relayOnline = false;
            break;
          }
        }
        this.windows.shift();
        savePending(this.windows, WINDOWS_KEY);
      }
    } finally {
      this.busyWin = false;
      this.emit();
    }
  }
}
