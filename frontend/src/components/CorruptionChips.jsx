import { CATEGORIES, CORRUPTIONS } from "../constants.js";

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
    <div>
      <p className="mb-2 text-xs font-medium text-slate-500">Corruption settings</p>
      <div className="flex flex-wrap gap-2">
        {entries.map(([key, value]) => {
          const [label, format] = FIELDS[key] ?? [key, (v) => String(v)];
          const colour =
            key === "type" && category
              ? `${category.soft} ${category.text} ${category.border}`
              : "border-slate-200 bg-slate-50 text-slate-700";
          return (
            <span key={key} className={`rounded-full border px-3 py-1 text-xs ${colour}`}>
              {label}: <span className={typeof value === "number" ? "font-mono" : "font-medium"}>{format(value)}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}
