import { useEffect, useMemo, useState } from "react";
import { API_BASE, GAZE_WINDOW_S, POLL, TIMELINE_HOURS, ZONES } from "./config.js";
import { liveApi } from "./api.js";
import { mockApi } from "./mock.js";
import { useNow, usePoll } from "./hooks.js";
import Header from "./components/Header.jsx";
import ConfusedBanner from "./components/ConfusedBanner.jsx";
import SuggestionCard from "./components/SuggestionCard.jsx";
import GazeHeatmap from "./components/GazeHeatmap.jsx";
import Timeline from "./components/Timeline.jsx";
import RequestFeed from "./components/RequestFeed.jsx";
import CareActions from "./components/CareActions.jsx";
import PatientState from "./components/PatientState.jsx";
import WebcamTrends, { TREND_MINUTES } from "./components/WebcamTrends.jsx";
import LearningPanel from "./components/LearningPanel.jsx";
import AlertBanner from "./components/AlertBanner.jsx";

// URL params: ?mock=1 (start in mock mode)  &api=http://host:8000  &patient=p1
const params = new URLSearchParams(window.location.search);
const apiBase = params.get("api") || API_BASE;

export default function App() {
  const [mode, setMode] = useState(params.get("mock") === "1" ? "mock" : "live");
  const [fellBack, setFellBack] = useState(false);
  const [patientId, setPatientId] = useState(params.get("patient"));
  const [confused, setConfused] = useState(false); // mock-mode demo switch
  const [nonce, setNonce] = useState(0); // bump to refetch immediately after a mock demo action
  const now = useNow(10000);

  const api = useMemo(() => (mode === "mock" ? mockApi : liveApi(apiBase)), [mode]);

  // ---- patients ----
  const patientsPoll = usePoll(() => api.getPatients(), POLL.patientsMs, [api]);
  const patients = patientsPoll.data ?? [];

  // Backend unreachable on first contact: fall back to mock data so the page still works.
  useEffect(() => {
    if (mode === "live" && patientsPoll.error && !patientsPoll.data) {
      setMode("mock");
      setFellBack(true);
    }
  }, [mode, patientsPoll.error, patientsPoll.data]);

  useEffect(() => {
    if (patientsPoll.data?.length && !patientsPoll.data.some((p) => p.id === patientId)) {
      setPatientId(patientsPoll.data[0].id);
    }
  }, [patientsPoll.data, patientId]);

  // ---- data for the selected patient ----
  const requestsPoll = usePoll(
    async () => {
      if (!patientId) return null;
      const to = new Date();
      const from = new Date(to.getTime() - TIMELINE_HOURS * 3600000);
      return api.getRequests(patientId, from, to);
    },
    POLL.requestsMs,
    [api, patientId, nonce]
  );
  const gazePoll = usePoll(
    async () => (patientId ? api.getGaze(patientId, GAZE_WINDOW_S) : null),
    POLL.gazeMs,
    [api, patientId, nonce]
  );
  // adaptive insights (falls back to POST /predict on an older backend)
  const predictPoll = usePoll(
    async () => (patientId ? api.getInsights(patientId) : null),
    POLL.predictMs,
    [api, patientId, nonce]
  );
  const windowsPoll = usePoll(
    async () => (patientId ? api.getWindows(patientId, TREND_MINUTES) : null),
    POLL.windowsMs,
    [api, patientId, nonce]
  );
  const statsPoll = usePoll(
    async () => (patientId ? api.getStats(patientId) : null),
    POLL.statsMs,
    [api, patientId, nonce]
  );
  const sendFeedback = async (label, action, suggestionId) => {
    await api.sendFeedback(patientId, label, action, suggestionId);
    setNonce((n) => n + 1);
  };

  // Keep the mock "confused" switch in step with the selected patient.
  useEffect(() => {
    if (mode === "mock" && patientId) setConfused(mockApi.isConfused(patientId));
  }, [mode, patientId]);

  const toggleConfused = () => {
    mockApi.setConfused(patientId, !confused);
    setConfused(!confused);
    setNonce((n) => n + 1);
  };
  const addRequest = () => {
    const label = ZONES[Math.floor(Math.random() * ZONES.length)].label;
    mockApi.addRequest(patientId, label);
    setNonce((n) => n + 1);
  };
  const switchMode = (m) => {
    setFellBack(false);
    setMode(m);
  };

  const connError = mode === "live" && [requestsPoll, gazePoll, predictPoll, windowsPoll].some((p) => p.error);
  const prediction = predictPoll.data;
  const patientName = patients.find((p) => p.id === patientId)?.name;

  return (
    <div className="min-h-screen">
      <Header
        patients={patients}
        patientId={patientId}
        onPatient={setPatientId}
        mode={mode}
        onMode={switchMode}
        connError={connError}
        confused={confused}
        onToggleConfused={toggleConfused}
        onAddRequest={addRequest}
      />

      {fellBack && (
        <div className="bg-amber-400/15 text-amber-100">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-2 text-sm">
            <span>
              Could not reach the backend at <code className="rounded bg-black/30 px-1">{apiBase}</code>. Showing
              mock data instead.
            </span>
            <button
              onClick={() => switchMode("live")}
              className="rounded-lg border border-amber-300/50 px-2 py-0.5 hover:bg-amber-300/10"
            >
              Retry live
            </button>
          </div>
        </div>
      )}

      <AlertBanner insights={prediction} patientName={patientName} />
      {prediction?.confused && <ConfusedBanner prediction={prediction} patientName={patientName} />}

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-4 p-4 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <SuggestionCard poll={predictPoll} now={now} />
        </div>
        <div className="lg:col-span-7">
          <CareActions poll={predictPoll} onFeedback={sendFeedback} />
        </div>
        <div className="lg:col-span-5">
          <PatientState poll={predictPoll} />
        </div>
        <div className="lg:col-span-7">
          <WebcamTrends poll={windowsPoll} now={Date.now()} />
        </div>
        <div className="lg:col-span-5">
          <GazeHeatmap poll={gazePoll} now={now} />
        </div>
        <div className="lg:col-span-7">
          <Timeline poll={requestsPoll} now={now} />
        </div>
        <div className="lg:col-span-5">
          <RequestFeed poll={requestsPoll} now={now} />
        </div>
        <div className="lg:col-span-7">
          <LearningPanel statsPoll={statsPoll} insightsPoll={predictPoll} />
        </div>
      </main>
    </div>
  );
}
