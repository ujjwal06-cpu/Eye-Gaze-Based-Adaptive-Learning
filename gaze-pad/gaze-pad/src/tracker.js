// Loads WebGazer from the local copy placed in /public by scripts/copy-webgazer.mjs.

let loading = null;

export function loadWebGazer() {
  if (window.webgazer) return Promise.resolve(window.webgazer);
  if (loading) return loading;
  loading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = `${import.meta.env.BASE_URL}webgazer.js`;
    s.async = true;
    s.onload = () =>
      window.webgazer
        ? resolve(window.webgazer)
        : reject(new Error("WebGazer script loaded but window.webgazer is missing"));
    s.onerror = () => {
      loading = null;
      reject(
        new Error(
          'Could not load /webgazer.js. Run "npm install" then "npm run copy-webgazer".'
        )
      );
    };
    document.head.appendChild(s);
  });
  return loading;
}

export function setCameraPreview(on) {
  const wg = window.webgazer;
  if (!wg) return;
  try {
    wg.showVideoPreview(!!on);
  } catch {
    /* preview is cosmetic */
  }
}

// Hide the built-in overlays; GazePad draws its own gaze dot.
export function quietWebGazer(wg) {
  const call = (name, arg) => {
    try {
      wg[name]?.(arg);
    } catch {
      /* older/newer versions may lack some of these */
    }
  };
  call("showPredictionPoints", false);
  call("showFaceOverlay", false);
  call("showFaceFeedbackBox", false);
}
