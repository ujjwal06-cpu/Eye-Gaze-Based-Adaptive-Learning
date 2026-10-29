/** Red bar for urgent situations other than confusion (an open Help/Pain request, distress). */
export default function AlertBanner({ insights, patientName }) {
  const a = insights?.alert;
  if (a?.level !== "urgent") return null;
  const top = insights.actions?.find((x) => x.priority === "high");
  return (
    <div role="alert" className="bg-red-700 text-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5">
        <span className="text-xl" aria-hidden="true">▲</span>
        <div className="text-lg font-extrabold leading-tight">{a.text}</div>
        {top?.reasons?.[0] && <div className="text-sm text-red-100">{patientName ? `${patientName}: ` : ""}{top.reasons[0]}</div>}
      </div>
    </div>
  );
}
