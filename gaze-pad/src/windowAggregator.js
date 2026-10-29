// Collects gaze samples, blinks and emergency selections, and emits one B1 message for every
// completed 10-second window aligned to the wall clock (:00, :10, :20 ...).
import {
  WINDOW_MS,
  buildSlots,
  computeWindowFeatures,
  formatTimeWindow,
  userIdFor,
} from "./b1Window.js";

/**
 * getContext()  -> { patientId, padW, padH, cmPerPx, viewDistCm, fixationDeg, tz }, read at window close
 * onWindow({ message, key, start, valid })   a finished, spec-shaped B1 message
 * onSkip({ start, valid })                   window dropped: fewer than minValid of 100 slots had gaze
 *
 * The window that is already running when the aggregator starts is skipped: a partial window
 * would give B1 numbers computed from less than 10 seconds of data.
 */
export function createWindowAggregator({
  getContext,
  onWindow,
  onSkip,
  minValid = 50,
  graceMs = 300,
  startedAt = Date.now(),
}) {
  const samples = []; // { t, x, y, rot } in epoch ms, ascending
  const blinks = []; // epoch ms
  const emergencies = []; // epoch ms
  let lastDone = Math.floor(startedAt / WINDOW_MS);

  const countIn = (arr, from, to) => arr.filter((t) => t >= from && t < to).length;

  function finalize(w) {
    const start = w * WINDOW_MS;
    const end = start + WINDOW_MS;
    // a little history before the window so slot 0 can reuse the previous sample
    const inWindow = samples.filter((s) => s.t >= start - 300 && s.t <= end);
    const slots = buildSlots(inWindow, start);
    const valid = slots.filter(Boolean).length;

    if (valid < minValid) {
      onSkip?.({ start, valid });
      return;
    }
    const ctx = getContext();
    const features = computeWindowFeatures(slots, {
      ...ctx,
      blinks: countIn(blinks, start, end),
      emergencies: countIn(emergencies, start, end),
    });
    onWindow?.({
      message: {
        User_ID: userIdFor(ctx.patientId),
        Time_Window: formatTimeWindow(start, ctx.tz),
        ...features,
      },
      key: `${ctx.patientId}|${start}`,
      start,
      valid,
    });
  }

  return {
    addSample: (s) => samples.push(s),
    addBlink: (t) => blinks.push(t),
    addEmergency: (t) => emergencies.push(t),

    /** Call a few times per second; emits every window that ended at least graceMs ago. */
    tick(now) {
      const ready = Math.floor((now - graceMs) / WINDOW_MS) - 1;
      for (let w = lastDone + 1; w <= ready; w++) finalize(w);
      if (ready > lastDone) {
        lastDone = ready;
        const keepFrom = (lastDone + 1) * WINDOW_MS - 300;
        while (samples.length && samples[0].t < keepFrom) samples.shift();
        while (blinks.length && blinks[0] < keepFrom + 300) blinks.shift();
        while (emergencies.length && emergencies[0] < keepFrom + 300) emergencies.shift();
      }
    },
  };
}
