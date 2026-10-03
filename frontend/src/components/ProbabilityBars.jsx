import { CATEGORIES, CLASS_ORDER } from "../constants.js";
import Card from "./Card.jsx";

// "Classifier routing" card for Hard-Routed Restoration.
// One row per class: label, bar in the category colour, percentage. The highest row is bold.
// `result` is the backend JSON ({probabilities, predicted, selected_expert, identity_bypass}) or null.
export default function ProbabilityBars({ result, loading }) {
  const probabilities = result?.probabilities;

  return (
    <Card title="Classifier routing">
      <div className={`flex flex-col gap-3 ${loading ? "animate-pulse" : ""}`}>
        {CLASS_ORDER.map((name) => {
          const category = CATEGORIES[name];
          const p = probabilities?.[name];
          const isTop = result && name === result.predicted;
          return (
            <div key={name} className="flex items-center gap-3 text-sm">
              <span className={`w-32 shrink-0 ${isTop ? "font-semibold text-slate-900" : "text-slate-600"}`}>
                {category.label}
              </span>
              <div className="h-3 flex-1 overflow-hidden rounded-full bg-slate-100">
                {p !== undefined && (
                  <div className={`h-full rounded-full ${category.bar}`} style={{ width: `${p * 100}%` }} />
                )}
              </div>
              <span className={`w-14 text-right font-mono ${isTop ? "font-semibold" : "text-slate-600"}`}>
                {p === undefined ? "—" : `${(p * 100).toFixed(1)}%`}
              </span>
            </div>
          );
        })}
      </div>

      {result && <RoutingDecision result={result} />}

      <p className="mt-4 text-xs text-slate-500">Identity bypass is used when the image is predicted clean.</p>
    </Card>
  );
}

// "Predicted corruption: Blur" badge and "Selected expert: Blur expert".
function RoutingDecision({ result }) {
  const predicted = CATEGORIES[result.predicted];
  const expert = result.identity_bypass
    ? "Identity bypass (no specialist run)"
    : CATEGORIES[result.selected_expert]?.expert ?? result.selected_expert;

  return (
    <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-slate-200 pt-4 text-sm">
      <p className="flex items-center gap-2 text-slate-600">
        Predicted corruption:
        <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${predicted.soft} ${predicted.text} ${predicted.border}`}>
          {predicted.label}
        </span>
      </p>
      <p className="text-slate-600">
        Selected expert: <span className="font-semibold text-slate-900">{expert}</span>
      </p>
    </div>
  );
}
