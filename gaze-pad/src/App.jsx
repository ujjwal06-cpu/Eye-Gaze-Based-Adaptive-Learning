import GazePad from "./GazePad.jsx";

// Optional URL params: ?patient=1&api=http://localhost:8000&mock=1&mockgaze=1
// B1: &b1=0 (off) &relay=http://localhost:8100 &screencm=30 &distcm=50 &tz=utc &debug=1
const params = new URLSearchParams(window.location.search);

export default function App() {
  return (
    <GazePad
      patientId={params.get("patient") || undefined}
      apiBase={params.get("api") || undefined}
      initialMode={params.get("mock") === "1" ? "mock" : undefined}
      sendMockGaze={params.get("mockgaze") === "1"}
      b1={params.get("b1") !== "0"}
      relayBase={params.get("relay") || undefined}
      screenWidthCm={Number(params.get("screencm")) || undefined}
      viewDistCm={Number(params.get("distcm")) || undefined}
      tz={params.get("tz") === "utc" ? "utc" : undefined}
      debug={params.get("debug") === "1"}
    />
  );
}
