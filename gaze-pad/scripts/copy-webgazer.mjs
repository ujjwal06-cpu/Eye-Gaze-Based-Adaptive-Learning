// Copies everything WebGazer needs at runtime into public/ so it is served locally
// (no CDN dependency, no bundler quirks). Runs before dev and build.
//   1. webgazer.js
//   2. the MediaPipe face mesh files (WebGazer 3.x loads them from /mediapipe/face_mesh/;
//      without them begin() fails with "t is not a function" and the camera never starts)
import { cpSync, copyFileSync, existsSync, mkdirSync } from "node:fs";

const wgSrc = "node_modules/webgazer/dist/webgazer.js";
const meshSrc = "node_modules/@mediapipe/face_mesh";

if (!existsSync(wgSrc) || !existsSync(meshSrc)) {
  console.warn('[copy-webgazer] webgazer / @mediapipe/face_mesh not found. Run "npm install" first.');
  process.exit(0);
}

mkdirSync("public/mediapipe/face_mesh", { recursive: true });
copyFileSync(wgSrc, "public/webgazer.js");
cpSync(meshSrc, "public/mediapipe/face_mesh", { recursive: true });
console.log("[copy-webgazer] webgazer.js and mediapipe/face_mesh copied to public/");
