import test from "node:test";
import assert from "node:assert/strict";
import { userIdFor, formatTimeWindow, buildSlots, computeWindowFeatures, SLOTS } from "./b1Window.js";
import { earFromLandmarks, headRotation, rotationAngleDeg, headStepDeg, BlinkDetector } from "./faceMetrics.js";
import { createWindowAggregator } from "./windowAggregator.js";

const near = (a, b, eps = 0.01) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);
const O = { padW: 1000, padH: 600, cmPerPx: 0.03, viewDistCm: 50 };
const slotsAt = (x, y, n = SLOTS) => Array.from({ length: n }, () => ({ x, y, rot: null }));

test("userIdFor", () => {
  assert.equal(userIdFor(1), "USER001");
  assert.equal(userIdFor("p12"), "USER012");
  assert.equal(userIdFor("1234"), "USER1234");
});

test("formatTimeWindow utc and rollover", () => {
  const t = Date.UTC(2026, 0, 1, 23, 59, 55);
  assert.equal(formatTimeWindow(t, "utc"), "23:59:55-00:00:05");
});

test("deviation is measured from the fixed screen centre; 100px off is 3.434 deg", () => {
  const centre = computeWindowFeatures(slotsAt(0.5, 0.5), O);
  assert.equal(centre.Mean_Gaze_Deviation, 0);
  assert.equal(centre.Gaze_Stability, 1);
  const off = computeWindowFeatures(slotsAt(0.6, 0.5), O); // +100 px
  near(off.Mean_Gaze_Deviation, 3.434, 0.01);
  near(off.Max_Gaze_Deviation, 3.434, 0.01);
});

test("fixations: one steady run is 10 s; a jump splits it", () => {
  const steady = computeWindowFeatures(slotsAt(0.2, 0.3), O);
  assert.equal(steady.Avg_Fixation_Duration, 10);
  assert.equal(steady.Max_Fixation_Duration, 10);
  const s = [...slotsAt(0.2, 0.3, 60), ...slotsAt(0.8, 0.8, 40)];
  const f = computeWindowFeatures(s, O);
  assert.equal(f.Max_Fixation_Duration, 6);
  assert.equal(f.Avg_Fixation_Duration, 5);
});

test("buildSlots: stale samples leave null slots", () => {
  const start = 10000;
  const slots = buildSlots([{ t: start + 50, x: 0.5, y: 0.5 }], start);
  assert.ok(slots[0] && slots[1] && slots[2]);
  assert.equal(slots[3], null); // >250 ms old
  assert.equal(slots.length, SLOTS);
});

test("EAR formula", () => {
  const pos = Array.from({ length: 468 }, () => [0, 0, 0]);
  const set = (idx, open) => {
    const [a, b, c, d, e, f] = idx;
    pos[a] = [0, 0, 0]; pos[d] = [10, 0, 0];
    pos[b] = [3, open, 0]; pos[f] = [3, -open, 0];
    pos[c] = [7, open, 0]; pos[e] = [7, -open, 0];
  };
  set([33, 160, 158, 133, 153, 144], 1.5);
  set([362, 385, 387, 263, 373, 380], 1.5);
  near(earFromLandmarks(pos), 0.3, 1e-9);
});

test("head rotation: 5 degree turn measures 5 degrees, no turn 0", () => {
  const mk = (deg) => {
    const pos = Array.from({ length: 468 }, () => [0, 0, 0]);
    const r = (deg * Math.PI) / 180;
    const rot = ([x, y, z]) => [x * Math.cos(r) + z * Math.sin(r), y, -x * Math.sin(r) + z * Math.cos(r)];
    pos[33] = rot([-3, 0, 0]); pos[263] = rot([3, 0, 0]);
    pos[10] = rot([0, -5, 0]); pos[152] = rot([0, 5, 1]);
    return headRotation(pos);
  };
  near(rotationAngleDeg(mk(0), mk(0)), 0, 1e-6);
  near(rotationAngleDeg(mk(0), mk(5)), 5, 0.05);
});

test("head movement is sqrt(dyaw^2+dpitch^2+droll^2): 3 deg yaw + 4 deg pitch = 5", () => {
  const R = (yaw, pitch, roll) => {
    const [y, p, r] = [yaw, pitch, roll].map((d) => (d * Math.PI) / 180);
    const [cy, sy, cp, sp, cr, sr] = [Math.cos(y), Math.sin(y), Math.cos(p), Math.sin(p), Math.cos(r), Math.sin(r)];
    return [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr];
  };
  near(headStepDeg(R(0, 0, 0), R(3, 4, 0)), 5, 0.01);
  near(headStepDeg(R(10, 5, 2), R(10, 5, 2)), 0, 1e-9);
  near(headStepDeg(R(179, 0, 0), R(-179, 0, 0)), 2, 0.01); // wraps across +-180
});

test("BlinkDetector: blink counted, long closure and tiny dip not", () => {
  const d = new BlinkDetector();
  let t = 0;
  const feed = (ear, ms) => {
    let hit = false;
    for (let i = 0; i < ms / 33; i++) hit = d.update(ear, (t += 33)) || hit;
    return hit;
  };
  feed(0.3, 1000);
  feed(0.1, 130);
  assert.equal(d.update(0.3, (t += 33)), true);
  assert.equal(d.count, 1);
  feed(0.1, 2000); // eyes shut
  d.update(0.3, (t += 33));
  assert.equal(d.count, 1);
  feed(0.3, 1000);
  feed(0.26, 200); // slight dip, above 0.7 x baseline
  d.update(0.3, (t += 33));
  assert.equal(d.count, 1);
});

test("aggregator skips first partial window and sparse windows, emits full ones", () => {
  const out = [];
  const skipped = [];
  const base = 1_000_000_000_000 - (1_000_000_000_000 % 10000); // window boundary
  const agg = createWindowAggregator({
    startedAt: base + 4000, // starts mid-window
    getContext: () => ({ patientId: 1, ...O, tz: "utc" }),
    onWindow: (w) => out.push(w),
    onSkip: (s) => skipped.push(s),
  });
  // samples every 50 ms for 30 s from base
  for (let t = base; t < base + 30000; t += 50) {
    // window 2 (base+20000..) has only 2 s of data
    if (t >= base + 20000 && t < base + 28000) continue;
    agg.addSample({ t, x: 0.5, y: 0.5, rot: null });
  }
  agg.addBlink(base + 12000);
  agg.addBlink(base + 15000);
  agg.addEmergency(base + 11000);
  agg.tick(base + 30400);
  assert.equal(out.length, 1, JSON.stringify(skipped));
  assert.equal(out[0].message.Blink_Rate, 2);
  assert.equal(out[0].message.Emergency_Button, 1);
  assert.equal(out[0].message.User_ID, "USER001");
  assert.equal(out[0].key, `1|${base + 10000}`);
  assert.equal(skipped.length, 1);
});
