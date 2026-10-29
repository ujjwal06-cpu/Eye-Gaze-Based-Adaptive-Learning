// Uplink to the FastAPI backend.
//  - gaze events: buffered, batched every FLUSH_MS, lossy if the backend is down for long
//  - request events: queued and persisted in localStorage, retried until delivered
import { FLUSH_MS } from "./config.js";

const PENDING_KEY = "gazepad.pendingRequests";
const MAX_GAZE_BUFFER = 5000;
const MAX_BATCH = 500;
const MAX_PENDING_REQUESTS = 200;

async function postJSON(url, body, timeoutMs = 4000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
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

function loadPending() {
  try {
    return JSON.parse(localStorage.getItem(PENDING_KEY) || "[]");
  } catch {
    return [];
  }
}
function savePending(list) {
  try {
    localStorage.setItem(PENDING_KEY, JSON.stringify(list));
  } catch {
    /* storage full or blocked: queue stays in memory only */
  }
}

export class Uplink {
  constructor({ apiBase, onStatus } = {}) {
    this.apiBase = apiBase;
    this.onStatus = onStatus;
    this.gaze = [];
    this.requests = loadPending();
    this.busyGaze = false;
    this.busyReq = false;
    this.online = null; // null = nothing sent yet
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

  status() {
    return { online: this.online, pendingGaze: this.gaze.length, pendingRequests: this.requests.length };
  }

  emit() {
    this.onStatus?.(this.status());
  }

  async flush() {
    await Promise.all([this.flushGaze(), this.flushRequests()]);
    this.emit();
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
}
