import test from "node:test";
import assert from "node:assert/strict";
import { zoneFromXY, DwellTracker } from "./gazeLogic.js";

test("zoneFromXY maps the 3x2 grid row-major", () => {
  assert.equal(zoneFromXY(0.1, 0.1), "water");
  assert.equal(zoneFromXY(0.5, 0.1), "food");
  assert.equal(zoneFromXY(0.9, 0.1), "help");
  assert.equal(zoneFromXY(0.1, 0.9), "pain");
  assert.equal(zoneFromXY(0.5, 0.9), "bathroom");
  assert.equal(zoneFromXY(0.9, 0.9), "rest");
});

test("zoneFromXY handles edges and out-of-range points", () => {
  assert.equal(zoneFromXY(1, 1), "rest");
  assert.equal(zoneFromXY(0, 0), "water");
  assert.equal(zoneFromXY(-0.01, 0.5), null);
  assert.equal(zoneFromXY(0.5, 1.2), null);
  assert.equal(zoneFromXY(null, 0.5), null);
  assert.equal(zoneFromXY(NaN, 0.5), null);
});

// Drive the tracker at 50 ms ticks.
function run(tracker, zone, fromMs, toMs, step = 50) {
  let last;
  const fired = [];
  for (let t = fromMs; t <= toMs; t += step) {
    last = tracker.update(zone, t);
    if (last.fired) fired.push({ t, zone: last.fired });
  }
  return { last, fired };
}

test("fires once after the dwell time, not before", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  const { fired, last } = run(tr, "water", 0, 3000);
  assert.equal(fired.length, 1);
  assert.equal(fired[0].zone, "water");
  assert.ok(fired[0].t >= 1500 && fired[0].t <= 1550, `fired at ${fired[0].t}`);
  assert.equal(last.armed, false);
  assert.equal(last.progress, 1);
});

test("does not fire for a glance shorter than the dwell time", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  const a = run(tr, "food", 0, 1200);
  const b = run(tr, "help", 1250, 2000);
  assert.equal(a.fired.length, 0);
  assert.equal(b.fired.length, 0);
});

test("a brief flicker into another zone does not reset the dwell", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  run(tr, "water", 0, 1000);
  run(tr, "food", 1050, 1150); // 150 ms flicker, below debounce
  const { fired } = run(tr, "water", 1200, 1700);
  assert.equal(fired.length, 1);
  assert.equal(fired[0].zone, "water");
  assert.ok(fired[0].t <= 1600, `fired at ${fired[0].t}`);
});

test("a real move to another zone restarts the dwell", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  run(tr, "water", 0, 1000);
  const { fired } = run(tr, "food", 1050, 2400);
  assert.equal(fired.length, 0); // food has only ~1.35 s
  const more = run(tr, "food", 2450, 3000);
  assert.equal(more.fired.length, 1);
  assert.equal(more.fired[0].zone, "food");
});

test("after firing, the same zone does not fire again until gaze leaves", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  const first = run(tr, "help", 0, 4000);
  assert.equal(first.fired.length, 1);
  run(tr, "rest", 4050, 4400); // leaves for > debounce
  const second = run(tr, "help", 4450, 6200);
  assert.equal(second.fired.length, 1);
});

test("losing gaze (null) clears the dwell", () => {
  const tr = new DwellTracker({ dwellMs: 1500, debounceMs: 250 });
  run(tr, "pain", 0, 1000);
  const { last } = run(tr, null, 1050, 1500);
  assert.equal(last.zone, null);
  assert.equal(last.dwellMs, 0);
});
