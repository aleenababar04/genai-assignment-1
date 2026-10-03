import { BRANCH_ORDER, CATEGORIES } from "../constants.js";
import Card from "./Card.jsx";

// Weights below this look "muted" on the expert cards.
const LOW_WEIGHT = 0.15;

// "Routing weights" card for Soft Mixture-of-Experts Restoration:
// a stacked contribution bar, a legend, and four expert cards with the top one highlighted.
// `result` is the backend JSON ({weights: {identity, salt_pepper, blur, occlusion}, top_branch}) or null.
export default function ExpertWeights({ result, loading }) {
  const weights = result?.weights;

  return (
    <Card title="Routing weights">
      {/* Stacked bar: each segment's width is the branch weight (weights sum to 1). */}
      <div className={`flex h-6 overflow-hidden rounded-full bg-slate-100 ${loading ? "animate-pulse" : ""}`}>
        {weights &&
          BRANCH_ORDER.map((name) => (
            <div
              key={name}
              className={`flex items-center justify-center font-mono text-xs font-medium text-white ${CATEGORIES[name].bar}`}
              style={{ width: `${weights[name] * 100}%` }}
              title={`${CATEGORIES[name].expert}: ${weights[name].toFixed(2)}`}
            >
              {weights[name] >= 0.1 ? weights[name].toFixed(2) : ""}
            </div>
          ))}
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        {BRANCH_ORDER.map((name) => (
          <span key={name} className="flex items-center gap-1.5">
            <span className={`size-2.5 rounded-sm ${CATEGORIES[name].bar}`} />
            {CATEGORIES[name].expert}
          </span>
        ))}
      </div>

      {/* Expert cards */}
      <div className="mt-5 grid grid-cols-2 gap-3 xl:grid-cols-4">
        {BRANCH_ORDER.map((name) => (
          <ExpertCard
            key={name}
            name={name}
            weight={weights?.[name]}
            isTop={result?.top_branch === name}
          />
        ))}
      </div>
    </Card>
  );
}

function ExpertCard({ name, weight, isTop }) {
  const category = CATEGORIES[name];
  const muted = weight !== undefined && !isTop && weight < LOW_WEIGHT;

  return (
    <div
      className={`rounded-lg p-3 ${isTop ? `border-2 ${category.border} ${category.soft}` : "border border-slate-200"} ${
        muted ? "opacity-50" : ""
      }`}
    >
      <div className="flex items-center gap-2">
        <span className={`size-2.5 shrink-0 rounded-full ${category.bar}`} />
        <span className="text-sm font-medium text-slate-700">{category.expert}</span>
      </div>
      <p className="mt-2 font-mono text-2xl font-semibold text-slate-900">
        {weight === undefined ? "—" : weight.toFixed(2)}
      </p>
      {isTop && (
        <span className={`mt-2 inline-block rounded-full px-2 py-0.5 text-xs font-semibold text-white ${category.bar}`}>
          Top contributor
        </span>
      )}
    </div>
  );
}
