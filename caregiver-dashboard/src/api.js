// Live API client. Every method returns normalised data (see utils.js), so the rest of
// the app never depends on the exact shape the backend chooses.
import {
  normalizePatients,
  normalizeRequests,
  normalizeGaze,
  normalizePrediction,
  normalizeInsights,
  normalizeWindows,
  normalizeStats,
} from "./utils.js";

async function http(url, options = {}, timeoutMs = 6000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: ctrl.signal });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status} for ${new URL(url).pathname}`);
      err.status = res.status;
      throw err;
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export function liveApi(base) {
  let hasInsights = true; // false once an older backend answers 404 for /insights

  const api = {
    kind: "live",
    base,

    async getPatients() {
      return normalizePatients(await http(`${base}/patients`));
    },

    async getRequests(patientId, from, to) {
      const qs = new URLSearchParams({
        patient_id: patientId,
        from: from.toISOString(),
        to: to.toISOString(),
      });
      return normalizeRequests(await http(`${base}/requests?${qs}`));
    },

    async getGaze(patientId, seconds) {
      const qs = new URLSearchParams({ patient_id: patientId, seconds: String(seconds) });
      return normalizeGaze(await http(`${base}/gaze/recent?${qs}`));
    },

    async predict(patientId, ts) {
      const body = JSON.stringify({ patient_id: patientId, ts });
      return normalizePrediction(
        await http(`${base}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        })
      );
    },

    /** Adaptive needs + webcam state + care actions. Falls back to /predict on an older backend. */
    async getInsights(patientId) {
      if (hasInsights) {
        try {
          const qs = new URLSearchParams({ patient_id: patientId });
          return normalizeInsights(await http(`${base}/insights?${qs}`, {}, 15000));
        } catch (e) {
          if (e.status !== 404 && e.status !== 405) throw e;
          hasInsights = false;
        }
      }
      return normalizeInsights(await api.predict(patientId, new Date().toISOString()));
    },

    async getWindows(patientId, minutes) {
      if (!hasInsights) return [];
      const qs = new URLSearchParams({ patient_id: patientId, minutes: String(minutes) });
      return normalizeWindows(await http(`${base}/windows?${qs}`));
    },

    async getStats(patientId) {
      if (!hasInsights) return null;
      return normalizeStats(await http(`${base}/model/stats?patient_id=${encodeURIComponent(patientId)}`, {}, 15000));
    },

    async sendFeedback(patientId, label, action, suggestionId) {
      return http(`${base}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: patientId, label, action, suggestion_id: suggestionId }),
      });
    },
  };
  return api;
}
