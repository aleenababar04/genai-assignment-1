import { formatMs } from "../constants.js";

// Compact "System status" card pinned to the bottom of the sidebar: backend health,
// how many ONNX models are loaded, and the last inference time (all from GET /api/health).
export default function SystemStatus({ health }) {
  const { status, data } = health;
  const lastMs = data?.last_inference_ms;
  const badge = {
    online: "border-emerald-200 bg-emerald-50 text-online",
    offline: "border-red-200 bg-red-50 text-red-600",
    checking: "border-line bg-slate-100 text-muted",
  }[status];

  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-line bg-page p-3">
      <div className="flex items-center justify-between gap-2">
        <StatusDot status={status} />
        <span className={`rounded border px-1.5 py-0.5 font-mono text-mono-sm font-medium ${badge}`}>
          {status === "online" ? "Healthy" : status === "offline" ? "Offline" : "…"}
        </span>
      </div>
      <div className="flex flex-col gap-1 border-t border-line pt-2 font-mono text-mono-sm">
        <div className="flex justify-between gap-2">
          <span className="text-muted">ONNX models:</span>
          {data ? (
            <span className="font-semibold whitespace-nowrap text-ink">
              {data.loaded_count} of {data.expected_count} loaded
            </span>
          ) : (
            <span className="text-muted">not loaded</span>
          )}
        </div>
        <div className="flex justify-between gap-2">
          <span className="text-muted">Last inference:</span>
          <span className={lastMs == null ? "text-muted" : "font-semibold whitespace-nowrap text-primary"}>
            {lastMs == null ? "none yet" : formatMs(lastMs)}
          </span>
        </div>
      </div>
    </div>
  );
}

// Green (pulsing) dot "Backend online", red dot "Backend offline", grey while checking.
export function StatusDot({ status, compact = false }) {
  const styles = {
    online: { dot: "bg-online", text: "text-ink", label: "Backend online" },
    offline: { dot: "bg-red-600", text: "text-red-700", label: "Backend offline" },
    checking: { dot: "bg-slate-400", text: "text-muted", label: "Checking backend..." },
  }[status];

  return (
    <p className={`flex items-center gap-2 font-semibold ${compact ? "text-body-sm" : "text-body-md"} ${styles.text}`}>
      <span className="relative flex size-2">
        {status === "online" && (
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-online opacity-75" />
        )}
        <span className={`relative inline-flex size-2 rounded-full ${styles.dot}`} />
      </span>
      {styles.label}
    </p>
  );
}
