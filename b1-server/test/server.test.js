import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

process.env.B1_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "b1s-"));
const { server } = await import("../server.js");
const { EXAMPLE, validate } = await import("../schema.js");

await new Promise((r) => server.listen(0, r));
const base = `http://127.0.0.1:${server.address().port}`;
const post = (env) => fetch(`${base}/window`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(env) });
const latest = async () => (await fetch(`${base}/latest`)).json();
const win = (user, startSec) => {
  const f = (s) => [Math.floor(s / 3600) % 24, Math.floor(s / 60) % 60, s % 60].map((n) => String(n).padStart(2, "0")).join(":");
  return { ...EXAMPLE, User_ID: user, Time_Window: `${f(startSec)}-${f(startSec + 10)}` };
};
test.after(() => server.close());

test("schema", () => {
  assert.deepEqual(validate(EXAMPLE), []);
  assert.ok(validate({ ...EXAMPLE, Blink_Rate: "3" }).length);
  assert.ok(validate({ ...EXAMPLE, Time_Window: "14:35:30-14:35:45" }).length);
  assert.deepEqual(validate({ ...EXAMPLE, Time_Window: "23:59:55-00:00:05" }), []);
});

test("empty list before any window", async () => assert.deepEqual(await latest(), []));

test("real window is served exactly in spec shape; same window stays until the next", async () => {
  assert.equal((await post({ message: win("USER001", 100), key: "1|100000", simulated: false })).status, 200);
  const a = await latest();
  assert.equal(a.length, 1);
  assert.deepEqual(Object.keys(a[0]).sort(), Object.keys(EXAMPLE).sort());
  assert.deepEqual(await latest(), a); // unchanged between polls
});

test("several patients -> list; older window never replaces newer", async () => {
  await post({ message: win("USER002", 100), key: "2|100000" });
  await post({ message: win("USER001", 120), key: "1|120000" });
  await post({ message: win("USER001", 110), key: "1|110000" }); // late arrival
  const l = await latest();
  assert.deepEqual(l.map((m) => [m.User_ID, m.Time_Window]), [["USER001", "00:02:00-00:02:10"], ["USER002", "00:01:40-00:01:50"]]);
});

test("invalid rejected with reasons and not exposed", async () => {
  const bad = win("USER009", 300);
  delete bad.Head_Movement;
  const r = await post({ message: bad, key: "9|300000" });
  assert.equal(r.status, 422);
  assert.match(JSON.stringify(await r.json()), /missing field 'Head_Movement'/);
  assert.ok(!(await latest()).some((m) => m.User_ID === "USER009"));
});

test("simulated (mouse) windows are held, not exposed", async () => {
  const r = await post({ message: win("USER007", 400), key: "7|400000", simulated: true });
  assert.equal((await r.json()).status, "held_simulated");
  assert.ok(!(await latest()).some((m) => m.User_ID === "USER007"));
});

test("/latest/:user and /status", async () => {
  assert.equal((await fetch(`${base}/latest/USER001`)).status, 200);
  assert.equal((await fetch(`${base}/latest/NOPE`)).status, 404);
  const s = await (await fetch(`${base}/status`)).json();
  assert.ok(s.polls > 0 && s.rejected === 1 && s.held_simulated === 1 && s.stale_ignored === 1);
});
