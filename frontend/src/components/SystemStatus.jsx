import { formatMs } from "../constants.js";

// Compact "System status" card for the sidebar: backend health, how many ONNX
// models are loaded, and the last inference time (all from GET /api/health).
export default function SystemStatus({ health }) {
  const { status, data } = health;
  const lastMs = data?.last_inference_ms;

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm">
      <p className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">System status</p>
      <StatusDot status={status} />
      <p className="mt-2 text-slate-600">
        ONNX models:{" "}
        {data ? (
          <span className="whitespace-nowrap text-slate-900">
            <span className="font-mono">{data.loaded_count}</span> of{" "}
            <span className="font-mono">{data.expected_count}</span> loaded
          </span>
        ) : (
          <span className="text-slate-500">not loaded</span>
        )}
      </p>
      <p className="mt-1 text-slate-600">
        Last inference:{" "}
        <span className={lastMs == null ? "text-slate-500" : "font-mono whitespace-nowrap text-slate-900"}>
          {lastMs == null ? "none yet" : formatMs(lastMs)}
        </span>
      </p>
    </div>
  );
}

// Green dot "Backend online", red dot "Backend offline", grey while checking.
export function StatusDot({ status }) {
  const styles = {
    online: { dot: "bg-green-600", text: "text-green-700", label: "Backend online" },
    offline: { dot: "bg-red-600", text: "text-red-700", label: "Backend offline" },
    checking: { dot: "bg-slate-400", text: "text-slate-500", label: "Checking backend..." },
  }[status];

  return (
    <p className={`flex items-center gap-2 font-medium ${styles.text}`}>
      <span className={`size-2.5 rounded-full ${styles.dot}`} />
      {styles.label}
    </p>
  );
}
