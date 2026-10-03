import { CATEGORIES, CORRUPTIONS } from "../constants.js";
import Icon from "./Icon.jsx";

// Shows the backend's corruption settings dict as a row of chips, e.g.
// {type: "blur", severity: "medium", kernel_size: 5, sigma: 1.5, seed: 42}
//   -> "Type: Gaussian blur" "Severity: Medium" "Kernel size: 5" "Sigma: 1.5" "Seed: 42"

// How each key is shown: [label, function that formats the value].
const FIELDS = {
  type: ["Type", (v) => CORRUPTIONS.find((c) => c.value === v)?.label ?? v],
  severity: ["Severity", (v) => v.charAt(0).toUpperCase() + v.slice(1)],
  probability: ["Noise probability", (v) => v],
  kernel_size: ["Kernel size", (v) => v],
  sigma: ["Sigma", (v) => v],
  num_rects: ["Rectangles", (v) => v],
  target_coverage: ["Target coverage", (v) => `${Math.round(v * 100)}%`],
  coverage: ["Actual coverage", (v) => `${(v * 100).toFixed(1)}%`],
  seed: ["Seed", (v) => v],
};
// Keys that are not shown as chips (the rectangle list is too long).
const HIDDEN = ["rectangles"];

export default function CorruptionChips({ settings }) {
  if (!settings) return null;
  const entries = Object.entries(settings).filter(([key]) => !HIDDEN.includes(key));
  // The "type" chip gets its category colour ("none" uses the clean colour).
  const category = CATEGORIES[settings.type === "none" ? "clean" : settings.type];

  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-line pt-4">
      <span className="pr-1 text-label-ui font-semibold tracking-wider text-muted uppercase">Corruption settings:</span>
      {entries.map(([key, value]) => {
        const [label, format] = FIELDS[key] ?? [key, (v) => String(v)];
        const isType = key === "type" && category;
        const colour = isType
          ? `${category.soft} ${category.text} ${category.softBorder}`
          : "border-line bg-slate-100 text-ink";
        return (
          <span key={key} className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-mono-sm ${colour}`}>
            {isType && <span className={`size-1.5 rounded-full ${category.bar}`} />}
            {key === "severity" && <Icon name="tune" size={12} className="text-muted" />}
            <span className={isType ? "" : "text-muted"}>{label}:</span>
            <span className={`font-semibold ${typeof value === "number" ? "text-primary" : ""}`}>{format(value)}</span>
          </span>
        );
      })}
    </div>
  );
}
