// Pure logic (no React, no DOM) so it can be unit tested with `npm test`.
import { ZONES, GRID_COLS, GRID_ROWS, DWELL_MS, DEBOUNCE_MS } from "./config.js";

export const clamp01 = (v) => Math.min(1, Math.max(0, v));

/**
 * Map a normalised point (0-1 relative to the pad) to a zone label.
 * Returns null if the point is outside the pad.
 */
export function zoneFromXY(x, y) {
  if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) return null;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  const col = Math.min(GRID_COLS - 1, Math.floor(x * GRID_COLS));
  const row = Math.min(GRID_ROWS - 1, Math.floor(y * GRID_ROWS));
  return ZONES[row * GRID_COLS + col].label;
}

/**
 * Debounced dwell tracker.
 *
 * - A new zone only becomes "active" after the gaze has stayed in it for
 *   `debounceMs`, so tracker jitter at zone borders does not reset the dwell.
 * - Dwell time is counted from the moment the gaze first entered the zone.
 * - A selection fires once when dwell reaches `dwellMs`, then the tracker is
 *   disarmed until the gaze moves to a different zone (or away).
 * - `dwellFor(zone)` (optional) gives a per-zone dwell time, e.g. a shorter dwell on the
 *   tile the adaptive model predicts. Falls back to `dwellMs`.
 */
export class DwellTracker {
  constructor({ dwellMs = DWELL_MS, debounceMs = DEBOUNCE_MS, dwellFor = null } = {}) {
    this.dwellMs = dwellMs;
    this.debounceMs = debounceMs;
    this.dwellFor = dwellFor;
    this.reset();
  }

  need(zone) {
    const v = this.dwellFor ? this.dwellFor(zone) : null;
    return Number.isFinite(v) && v > 0 ? v : this.dwellMs;
  }

  reset() {
    this.cand = { zone: null, since: 0 };
    this.active = { zone: null, since: 0 };
    this.armed = true;
  }

  update(zone, now) {
    if (zone !== this.cand.zone) this.cand = { zone, since: now };

    if (
      this.cand.zone !== this.active.zone &&
      now - this.cand.since >= this.debounceMs
    ) {
      this.active = { zone: this.cand.zone, since: this.cand.since };
      this.armed = true;
    }

    const dwell = this.active.zone ? now - this.active.since : 0;
    const need = this.active.zone ? this.need(this.active.zone) : this.dwellMs;
    let fired = null;
    if (this.active.zone && this.armed && dwell >= need) {
      fired = this.active.zone;
      this.armed = false;
    }
    return {
      zone: this.active.zone,
      dwellMs: dwell,
      progress: Math.min(1, dwell / need),
      armed: this.armed,
      fired,
    };
  }
}

/**
 * Adaptive dwell: the tile the model thinks is most likely (prob >= minProb) needs less dwell.
 * Never below floorMs, never changes other tiles, so a wrong prediction only costs normal speed.
 */
export function adaptiveDwell(baseMs, predictions, { factor = 0.75, minProb = 0.4, floorMs = 1000 } = {}) {
  const top = predictions?.[0];
  const fast = Math.max(floorMs, Math.round(baseMs * factor));
  return (zone) => (top && zone === top.label && top.prob >= minProb && fast < baseMs ? fast : baseMs);
}

/** Labels to highlight on the pad: the top predictions above minProb (at most `max`). */
export function hintLabels(predictions, { minProb = 0.3, max = 2 } = {}) {
  return (predictions || []).filter((p) => p.prob >= minProb).slice(0, max).map((p) => p.label);
}
