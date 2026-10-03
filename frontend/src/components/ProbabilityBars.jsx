import { CATEGORIES, CLASS_ORDER } from "../constants.js";
import Card from "./Card.jsx";
import Icon from "./Icon.jsx";

// "Classifier Routing & Dispatch" card for Hard-Routed Restoration.
// One row per class: colour dot, label, percentage and a bar. The predicted row is
// highlighted in its category colour with a "ROUTED" tag.
// `result` is the backend JSON ({probabilities, predicted, selected_expert, identity_bypass}) or null.
export default function ProbabilityBars({ result, loading }) {
  const probabilities = result?.probabilities;

  return (
    <Card title="Classifier Routing & Dispatch" subtitle="Softmax probability distribution across the four corruption classes">
      <div className={`flex flex-col gap-3.5 ${loading ? "animate-pulse" : ""}`}>
        {CLASS_ORDER.map((name) => {
          const category = CATEGORIES[name];
          const p = probabilities?.[name];
          const isTop = Boolean(result) && name === result.predicted;
          const width = p === undefined ? 0 : p * 100;
          return (
            <div
              key={name}
              className={`flex flex-col gap-1 ${isTop ? `-mx-2 rounded-lg border p-2 ${category.soft} ${category.softBorder}` : ""}`}
            >
              <div className="flex items-center justify-between font-mono text-mono-sm">
                <div className="flex items-center gap-2">
                  <span className={`size-2.5 rounded-full ${category.bar}`} />
                  <span className={isTop ? `font-bold ${category.text}` : "font-medium text-ink"}>{category.label}</span>
                  {isTop && (
                    <span className={`rounded px-1.5 text-[9px] font-bold tracking-wider text-white uppercase ${category.bar}`}>
                      Routed
                    </span>
                  )}
                </div>
                <span className={isTop ? `font-bold ${category.text}` : "text-muted"}>
                  {p === undefined ? "—" : `${(p * 100).toFixed(1)}%`}
                </span>
              </div>
              <div
                className={`w-full overflow-hidden rounded-full border ${
                  isTop ? `h-2.5 ${category.track} ${category.softBorder}` : "h-2 border-line bg-slate-100"
                }`}
              >
                <div className={`h-full rounded-full transition-all duration-500 ${category.bar}`} style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      {result && <RoutingDecision result={result} />}

      <p className="text-body-sm text-muted italic">Identity bypass is used when the image is predicted clean.</p>
    </Card>
  );
}

// "Predicted corruption: Blur" and "Selected expert: Blur expert" pills,
// plus the classifier's confidence (the highest probability).
function RoutingDecision({ result }) {
  const predicted = CATEGORIES[result.predicted];
  const expert = result.identity_bypass
    ? "Identity bypass (no specialist run)"
    : CATEGORIES[result.selected_expert]?.expert ?? result.selected_expert;
  const confidence = result.probabilities?.[result.predicted];

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-label-ui font-semibold ${predicted.soft} ${predicted.text} ${predicted.softBorder}`}>
          <span className={`size-2 rounded-full ${predicted.bar}`} />
          Predicted corruption: {predicted.label}
        </span>
        <span className="flex items-center gap-1 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-label-ui font-semibold text-primary">
          <Icon name="check_circle" size={15} />
          Selected expert: {expert}
        </span>
      </div>
      {confidence !== undefined && (
        <span className="font-mono text-mono-sm text-muted">
          Confidence score: <span className={`font-semibold ${predicted.accent}`}>{confidence.toFixed(3)}</span>
        </span>
      )}
    </div>
  );
}
