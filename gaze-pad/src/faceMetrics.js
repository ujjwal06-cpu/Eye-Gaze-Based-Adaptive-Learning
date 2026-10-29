// Blink and head-rotation measurements from WebGazer's face mesh points.
// WebGazer exposes them as webgazer.getTracker().getPositions(): an array of [x, y, z]
// points (MediaPipe face mesh, 468+ landmarks, x/y in video pixels, z in the same scale).

// MediaPipe face mesh landmark indices for the six points of each eye (p1..p6).
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
// Face frame: eye outer corners, forehead, chin.
const EYE_L = 33;
const EYE_R = 263;
const FOREHEAD = 10;
const CHIN = 152;

const RAD = 180 / Math.PI;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0)];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export const hasLandmarks = (pos) => Array.isArray(pos) && pos.length >= 468;

/** Eye aspect ratio (open eye about 0.25-0.35, closed below about 0.15), averaged over both eyes. */
export function earFromLandmarks(pos) {
  const one = (idx) => {
    const [p1, p2, p3, p4, p5, p6] = idx.map((i) => pos[i]);
    const width = d2(p1, p4);
    return width > 0 ? (d2(p2, p6) + d2(p3, p5)) / (2 * width) : 0;
  };
  return (one(RIGHT_EYE) + one(LEFT_EYE)) / 2;
}

/**
 * Head orientation as a 3x3 rotation matrix (row-major, 9 numbers) built from the face frame:
 * x axis = across the eyes, y axis = forehead to chin, z axis = out of the face.
 * Only differences between two of these matter (see rotationAngleDeg), so no axis convention needed.
 */
export function headRotation(pos) {
  const xa = unit(sub(pos[EYE_R], pos[EYE_L]));
  const za = unit(cross(xa, sub(pos[CHIN], pos[FOREHEAD])));
  const ya = cross(za, xa);
  return [xa[0], ya[0], za[0], xa[1], ya[1], za[1], xa[2], ya[2], za[2]];
}

/** Angle in degrees between two head orientations (geodesic distance between rotations). */
export function rotationAngleDeg(a, b) {
  let tr = 0; // trace(A^T B) = sum of the 9 element-wise products
  for (let i = 0; i < 9; i++) tr += a[i] * b[i];
  const c = (tr - 1) / 2;
  return Math.acos(Math.max(-1, Math.min(1, c))) * RAD;
}

/** Yaw, pitch, roll in degrees from a head rotation matrix (ZYX convention). */
export function eulerDeg(m) {
  const pitch = Math.asin(Math.max(-1, Math.min(1, -m[6])));
  const yaw = Math.atan2(m[3], m[0]);
  const roll = Math.atan2(m[7], m[8]);
  return [yaw * RAD, pitch * RAD, roll * RAD];
}

const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

/** Team B's head movement: sqrt(dyaw^2 + dpitch^2 + droll^2) between two samples, in degrees. */
export function headStepDeg(a, b) {
  const [y1, p1, r1] = eulerDeg(a);
  const [y2, p2, r2] = eulerDeg(b);
  return Math.hypot(wrap(y2 - y1), wrap(p2 - p1), wrap(r2 - r1));
}

/**
 * Counts complete blinks from a stream of eye aspect ratios.
 * A blink = the ratio drops below closeRatio x the person's own open-eye baseline for
 * minClosedMs..maxClosedMs, then recovers. Longer closures are eyes shut, not a blink.
 */
export class BlinkDetector {
  constructor({ closeRatio = 0.7, minClosedMs = 60, maxClosedMs = 1000 } = {}) {
    this.closeRatio = closeRatio;
    this.minClosedMs = minClosedMs;
    this.maxClosedMs = maxClosedMs;
    this.baseline = null;
    this.closedSince = null;
    this.count = 0;
  }

  /** Feed one frame. Returns true when a blink has just been completed. */
  update(ear, tMs) {
    if (this.baseline === null) {
      this.baseline = ear;
      return false;
    }
    if (ear >= this.baseline * this.closeRatio) {
      this.baseline += 0.05 * (ear - this.baseline); // adapt only while the eye is open
      const since = this.closedSince;
      this.closedSince = null;
      if (since !== null) {
        const dur = tMs - since;
        if (dur >= this.minClosedMs && dur <= this.maxClosedMs) {
          this.count += 1;
          return true;
        }
      }
      return false;
    }
    if (this.closedSince === null) this.closedSince = tMs;
    else if (tMs - this.closedSince > this.maxClosedMs) {
      // low for too long: treat as a new normal (squint, head tilt), not a blink
      this.baseline = ear;
      this.closedSince = null;
    }
    return false;
  }

  /** Face left the frame: forget a half-finished closure so it is never counted. */
  faceLost() {
    this.closedSince = null;
  }
}
