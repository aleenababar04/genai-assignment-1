import Icon from "./Icon.jsx";

// Amber notice listing the ONNX files a workspace needs that the backend has not loaded.
// The workspace still works for the parts that do not need them; the backend
// answers 503 for the rest, which the page shows in an ErrorCard.
export default function ModelNotice({ health, required }) {
  if (health.status !== "online" || !health.data) return null;
  const missing = required.filter((name) => !health.data.models?.[name]?.loaded);
  if (missing.length === 0) return null;

  return (
    <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-body-md text-amber-800">
      <Icon name="warning" size={20} className="mt-px text-amber-500" />
      <p>
        <span className="font-semibold">Model not loaded: </span>
        <span className="font-mono text-mono-lg">{missing.join(", ")}</span>
        <span>. Requests that need it will fail until the file is placed in the models folder and the backend is restarted.</span>
      </p>
    </div>
  );
}
