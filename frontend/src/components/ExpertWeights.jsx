import { BRANCH_ORDER, CATEGORIES } from "../constants.js";
import Card from "./Card.jsx";

// Weights below this look "muted" on the expert cards.
const LOW_WEIGHT = 0.15;

// Short legend names under the stacked bar.
const SHORT = { identity: "Clean", salt_pepper: "S&P", blur: "Blur", occlusion: "Occlusion" };

// "Gating Network Routing Weights" card for Soft Mixture-of-Experts Restoration:
// a stacked contribution bar with a legend, four expert cards (top one highlighted),
// and readouts computed from the weights (sum and entropy).
// `result` is the backend JSON ({weights: {identity, salt_pepper, blur, occlusion}, top_branch}) or null.
export default function ExpertWeights({ result, loading }) {
  const weights = result?.weights;
  const total = weights ? BRANCH_ORDER.reduce((sum, name) => sum + weights[name], 0) : null;
  // Shannon entropy in bits: 0 = one expert does everything, 2 = all four equal.
  const entropy = weights
    ? -BRANCH_ORDER.reduce((sum, name) => (weights[name] > 0 ? sum + weights[name] * Math.log2(weights[name]) : sum), 0)
    : null;

  return (
    <Card
      title="Gating Network Routing Weights"
      subtitle="Softmax contribution of each of the 4 expert branches"
      aside={
        <span className="rounded bg-slate-100 px-2.5 py-1 font-mono text-mono-sm text-ink">
          Total: {total === null ? "—" : `${total.toFixed(3)} (100%)`}
        </span>
      }
    >
      {/* Stacked bar: each segment's width is the branch weight (weights sum to 1). */}
      <div className="flex flex-col gap-1.5">
        <div className={`flex h-4 w-full overflow-hidden rounded-md bg-slate-100 shadow-inner ${loading ? "animate-pulse" : ""}`}>
          {weights &&
            BRANCH_ORDER.map((name) => (
              <div
                key={name}
                className={`h-full transition-all ${CATEGORIES[name].bar}`}
                style={{ width: `${weights[name] * 100}%` }}
                title={`${CATEGORIES[name].expert}: ${weights[name].toFixed(2)}`}
              />
            ))}
        </div>
        <div className="flex flex-wrap justify-between gap-x-4 px-0.5 font-mono text-mono-sm">
          {BRANCH_ORDER.map((name) => {
            const isTop = result?.top_branch === name;
            return (
              <span key={name} className={`flex items-center gap-1.5 ${CATEGORIES[name].accent} ${isTop ? "font-bold" : "font-medium"}`}>
                {!weights && <span className={`size-2 rounded-full ${CATEGORIES[name].bar}`} />}
                {weights ? `${Math.round(weights[name] * 100)}% ` : ""}
                {SHORT[name]}
                {isTop ? " (dominant)" : ""}
              </span>
            );
          })}
        </div>
      </div>

      {/* Expert cards */}
      <div className="grid grid-cols-2 gap-3 pt-1 xl:grid-cols-4">
        {BRANCH_ORDER.map((name) => (
          <ExpertCard key={name} name={name} weight={weights?.[name]} isTop={result?.top_branch === name} />
        ))}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3 font-mono text-mono-sm text-muted">
        <span>
          Entropy: <strong className="text-ink">{entropy === null ? "—" : `${entropy.toFixed(2)} bits`}</strong>
          <span className="text-slate-400"> (max 2.00)</span>
        </span>
        <span className="flex items-center gap-1.5 font-medium text-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 14 }} aria-hidden="true">tune</span>
          Output = weighted sum of the 4 expert outputs
        </span>
      </div>
    </Card>
  );
}

function ExpertCard({ name, weight, isTop }) {
  const category = CATEGORIES[name];
  const muted = weight !== undefined && !isTop && weight < LOW_WEIGHT;

  return (
    <div
      className={`relative flex flex-col justify-between rounded-lg p-3 ${
        isTop ? `border-2 ${category.border} ${category.soft} shadow-xs` : "border border-line bg-page"
      } ${muted ? "opacity-60" : ""}`}
    >
      {isTop && (
        <span className={`absolute -top-2.5 right-2 rounded-full px-1.5 py-0.5 font-mono text-[9px] font-bold tracking-wide text-white uppercase shadow-xs ${category.bar}`}>
          Top contributor
        </span>
      )}
      <div>
        <div className="mb-1 flex items-center gap-1.5">
          <span className={`size-2 shrink-0 rounded-full ${category.bar}`} />
          <span className={`truncate text-label-ui ${isTop ? `font-semibold ${category.accent}` : "text-muted"}`}>{category.expert}</span>
        </div>
        <p className="font-mono text-mono-lg font-bold text-ink">
          {weight === undefined ? "—" : weight.toFixed(2)}{" "}
          {weight !== undefined && (
            <span className={`font-sans text-body-sm font-normal ${isTop ? category.accent : "text-muted"}`}>
              ({Math.round(weight * 100)}%)
            </span>
          )}
        </p>
      </div>
      <div className={`mt-2 h-1 w-full overflow-hidden rounded-full ${isTop ? category.track : "bg-slate-200"}`}>
        <div className={`h-full ${category.bar}`} style={{ width: `${(weight ?? 0) * 100}%` }} />
      </div>
    </div>
  );
}
