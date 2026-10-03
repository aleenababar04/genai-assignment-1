import { formatMs } from "../constants.js";

// "Inference time 38 ms" tile. Shows a grey skeleton bar while loading and a dash when empty.
export default function StatTile({ label = "Inference time", ms, loading }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
      <p className="text-xs text-slate-500">{label}</p>
      {loading ? (
        <div className="mt-1.5 h-6 w-20 animate-pulse rounded bg-slate-200" />
      ) : (
        <p className="mt-0.5 font-mono text-xl font-semibold text-slate-900">{formatMs(ms)}</p>
      )}
    </div>
  );
}
