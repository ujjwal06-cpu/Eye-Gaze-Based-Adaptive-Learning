import GazePad from "./GazePad.jsx";

// Optional URL params: ?patient=p1&api=http://localhost:8000&mock=1&mockgaze=1
const params = new URLSearchParams(window.location.search);

export default function App() {
  return (
    <GazePad
      patientId={params.get("patient") || undefined}
      apiBase={params.get("api") || undefined}
      initialMode={params.get("mock") === "1" ? "mock" : undefined}
      sendMockGaze={params.get("mockgaze") === "1"}
    />
  );
}
