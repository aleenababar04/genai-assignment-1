import { formatMs } from "../constants.js";

// "INFERENCE TIME 38 ms / frame" tile. Shows a grey skeleton bar while loading and a dash when empty.
export default function StatTile({ label = "Inference time", ms, loading }) {
  const [value, unit] = ms === null || ms === undefined ? ["—", "ms"] : formatMs(ms).split(" ");
  return (
    <div className="flex flex-col justify-between rounded-lg border border-line bg-page p-3">
      <span className="font-mono text-mono-sm text-muted uppercase">{label}</span>
      {loading ? (
        <div className="mt-1.5 h-6 w-24 animate-pulse rounded bg-slate-200" />
      ) : (
        <div className="mt-1 flex items-baseline gap-1">
          <span className={`font-mono text-headline-md ${ms == null ? "text-slate-400" : "text-primary"}`}>
            {value} {unit}
          </span>
          <span className="font-mono text-mono-sm text-muted">/ image</span>
        </div>
      )}
    </div>
  );
}

// A second tile with a label and a short mono text (e.g. the model file).
export function InfoTile({ label, children }) {
  return (
    <div className="flex min-w-0 flex-col justify-between rounded-lg border border-line bg-page p-3">
      <span className="font-mono text-mono-sm text-muted uppercase">{label}</span>
      <div className="mt-1 font-mono text-mono-md font-semibold break-words text-ink">{children}</div>
    </div>
  );
}
