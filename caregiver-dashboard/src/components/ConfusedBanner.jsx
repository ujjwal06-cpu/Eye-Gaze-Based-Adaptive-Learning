import { META } from "../config.js";
import { pct } from "../utils.js";

export default function ConfusedBanner({ prediction, patientName }) {
  const top = prediction?.suggestions?.[0];
  const meta = top ? META[top.label] : null;
  return (
    <div role="alert" className="bg-red-600 text-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-1 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl" aria-hidden="true">
            {"⚠️"}
          </span>
          <div>
            <div className="text-xl font-extrabold leading-tight">Patient may be confused</div>
            <div className="text-sm text-red-100">
              {patientName ? `${patientName}: ` : ""}gaze is scattered (confusion score{" "}
              {prediction.confusion_score.toFixed(2)}). Please check on them.
            </div>
          </div>
        </div>
        {top && (
          <div className="ml-auto rounded-xl bg-white/15 px-4 py-2 text-sm">
            Most likely need:{" "}
            <strong>
              {meta?.icon} {meta?.text ?? top.label} ({pct(top.prob)})
            </strong>
          </div>
        )}
      </div>
    </div>
  );
}
