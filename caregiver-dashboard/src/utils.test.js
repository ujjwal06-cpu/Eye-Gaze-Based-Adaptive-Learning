import test from "node:test";
import assert from "node:assert/strict";
import {
  parseTs,
  zoneFromXY,
  normalizePatients,
  normalizeRequests,
  normalizeGaze,
  normalizePrediction,
  timeAgo,
} from "./utils.js";

test("parseTs treats naive ISO strings as UTC", () => {
  const z = Date.parse("2026-10-17T03:34:35.161Z");
  assert.equal(parseTs("2026-10-17T03:34:35.161"), z);
  assert.equal(parseTs("2026-10-17 03:34:35.161"), z);
  assert.equal(parseTs("2026-10-17T03:34:35.161Z"), z);
  assert.equal(parseTs("2026-10-17T09:04:35.161+05:30"), z);
  assert.ok(Number.isNaN(parseTs(undefined)));
});

test("zoneFromXY is row-major over 3x2", () => {
  assert.equal(zoneFromXY(0.1, 0.1), "water");
  assert.equal(zoneFromXY(0.9, 0.1), "help");
  assert.equal(zoneFromXY(0.5, 0.9), "bathroom");
  assert.equal(zoneFromXY(1, 1), "rest");
  assert.equal(zoneFromXY(1.2, 0.5), null);
});

test("normalizePatients accepts strings, objects and wrapped lists", () => {
  assert.deepEqual(normalizePatients(["p1", "p2"]), [
    { id: "p1", name: "p1" },
    { id: "p2", name: "p2" },
  ]);
  assert.deepEqual(normalizePatients({ patients: [{ id: 7, name: "Asha" }] }), [{ id: "7", name: "Asha" }]);
  assert.deepEqual(normalizePatients([{ patient_id: "x" }]), [{ id: "x", name: "x" }]);
  assert.deepEqual(normalizePatients(null), []);
});

test("normalizeRequests parses timestamps and drops bad rows", () => {
  const out = normalizeRequests({
    items: [
      { patient_id: "p1", ts: "2026-10-17T03:34:35", request: "water", source: "gaze" },
      { patient_id: "p1", ts: "garbage", request: "food" },
      { patient_id: "p1", ts: "2026-10-17T03:34:35Z" },
    ],
  });
  assert.equal(out.length, 1);
  assert.equal(out[0].request, "water");
  assert.equal(out[0].t, Date.parse("2026-10-17T03:34:35Z"));
});

test("normalizeGaze derives a missing zone from x,y but keeps an explicit null", () => {
  const out = normalizeGaze([
    { ts: "2026-10-17T03:34:35Z", x: 0.1, y: 0.1 },
    { ts: "2026-10-17T03:34:36Z", x: 0.1, y: 0.1, zone: null },
    { ts: "2026-10-17T03:34:37Z", x: 0.9, y: 0.9, zone: "rest", dwell_ms: 300, confidence: 0.8 },
  ]);
  assert.equal(out[0].zone, "water");
  assert.equal(out[1].zone, null);
  assert.equal(out[2].dwell_ms, 300);
});

test("normalizePrediction sorts, keeps the top 3 and coerces numbers", () => {
  const p = normalizePrediction({
    suggestions: [
      { label: "food", prob: "0.1" },
      { label: "water", prob: 0.6 },
      { label: "rest", prob: 0.2 },
      { label: "pain", prob: 0.05 },
    ],
    confused: true,
    confusion_score: "0.83",
  });
  assert.deepEqual(p.suggestions.map((s) => s.label), ["water", "rest", "food"]);
  assert.equal(p.confused, true);
  assert.equal(p.confusion_score, 0.83);
  assert.deepEqual(normalizePrediction({}), { suggestions: [], confused: false, confusion_score: 0 });
});

test("timeAgo buckets", () => {
  const now = 1_000_000_000;
  assert.equal(timeAgo(now - 3000, now), "just now");
  assert.equal(timeAgo(now - 42000, now), "42s ago");
  assert.equal(timeAgo(now - 5 * 60000, now), "5 min ago");
  assert.equal(timeAgo(now - 3 * 3600000, now), "3 h ago");
});

import { normalizeInsights, normalizeWindows, normalizeStats } from "./utils.js";

test("normalizeInsights reads the adaptive response and falls back to a /predict response", () => {
  const full = normalizeInsights({
    needs: [{ label: "food", prob: 0.2, reasons: [] }, { label: "water", prob: 0.7, reasons: ["Usually at 10:00"] }],
    suggestions: [{ label: "water", prob: 0.7 }],
    confused: false, confusion_score: 0.1,
    alert: { level: "attention", text: "Offer water" },
    actions: [{ id: "need:water", label: "water", priority: "medium", title: "Offer water", prob: 0.7, reasons: ["x"] }],
    webcam: { available: true, states: { distress: 0.2, fatigue: 0.1, confusion: 0.1 } },
    last_request: { t: "2026-10-17T03:34:35Z", label: "food" },
  });
  assert.equal(full.adaptive, true);
  assert.equal(full.needs[0].label, "water");
  assert.equal(full.actions[0].title, "Offer water");
  assert.equal(full.lastRequest.t, Date.parse("2026-10-17T03:34:35Z"));
  const old = normalizeInsights({ suggestions: [{ label: "rest", prob: 0.5 }], confused: true, confusion_score: 0.9 });
  assert.equal(old.adaptive, false);
  assert.equal(old.needs[0].label, "rest");
  assert.equal(old.webcam.available, false);
  assert.deepEqual(old.actions, []);
});

test("normalizeWindows flattens the B1 message and sorts by time", () => {
  const w = normalizeWindows([
    { start: "2026-10-17T03:34:45Z", message: { Head_Movement: 2, Gaze_Stability: 0.4 } },
    { start: "2026-10-17T03:34:35Z", message: { Head_Movement: 1, Gaze_Stability: 0.3 } },
  ]);
  assert.equal(w.length, 2);
  assert.equal(w[0].Head_Movement, 1);
  assert.ok(w[0].t < w[1].t);
});

test("normalizeStats tolerates missing fields", () => {
  assert.equal(normalizeStats(null), null);
  const s = normalizeStats({ n: 3, rolling: [{ i: 1, ts: "2026-10-17T03:34:35Z", top1: 1, top3: 1 }] });
  assert.equal(s.n, 3);
  assert.equal(s.rolling[0].top1, 1);
});
