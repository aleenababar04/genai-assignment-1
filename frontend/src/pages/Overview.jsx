import { PAGES, formatMs } from "../constants.js";
import PageHeader from "../components/PageHeader.jsx";
import Card from "../components/Card.jsx";
import { StatusDot } from "../components/SystemStatus.jsx";

// One card per workspace on the Overview page.
const WORKSPACES = [
  { id: "universal", icon: "U", description: "One autoencoder restores any corruption." },
  { id: "hard-routed", icon: "H", description: "A classifier detects the corruption and routes the image to one specialist." },
  { id: "soft-moe", icon: "S", description: "A gating network blends four expert branches." },
  { id: "face-to-sketch", icon: "F", description: "A conditional GAN turns a face photo into a sketch in three styles." },
];

export default function Overview({ health }) {
  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle="Four workspaces for restoring corrupted images and generating sketches from face photos."
      />

      <div className="grid gap-6 md:grid-cols-2">
        {WORKSPACES.map((workspace) => (
          <Card key={workspace.id}>
            <div className="flex h-full flex-col gap-3">
              <span className="flex size-10 items-center justify-center rounded-lg bg-indigo-50 font-semibold text-indigo-600">
                {workspace.icon}
              </span>
              <h2 className="text-base font-semibold">{PAGES.find((p) => p.id === workspace.id).title}</h2>
              <p className="flex-1 text-sm text-slate-500">{workspace.description}</p>
              <a
                href={`#/${workspace.id}`}
                className="self-start rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
              >
                Open workspace
              </a>
            </div>
          </Card>
        ))}
      </div>

      <SystemCard health={health} />
    </div>
  );
}

// Full-width "System" card: backend status, one chip per ONNX model, last inference time.
function SystemCard({ health }) {
  const models = health.data?.models ?? {};
  const lastMs = health.data?.last_inference_ms;

  return (
    <Card title="System" className="mt-6">
      <StatusDot status={health.status} />

      <p className="mt-4 mb-2 text-xs font-medium text-slate-500">ONNX models</p>
      {Object.keys(models).length === 0 ? (
        <p className="text-sm text-slate-500">No model information (backend not reachable).</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {Object.entries(models).map(([name, info]) => (
            <span
              key={name}
              title={info.error ?? "Loaded"}
              className={`rounded-full border px-3 py-1 text-xs ${
                info.loaded
                  ? "border-green-200 bg-green-50 text-green-700"
                  : "border-red-200 bg-red-50 text-red-700"
              }`}
            >
              <span className="font-mono">{name}</span> · {info.loaded ? "Loaded" : "Not loaded"}
            </span>
          ))}
        </div>
      )}

      <p className="mt-4 text-sm text-slate-600">
        Last inference:{" "}
        <span className="font-mono text-slate-900">{lastMs == null ? "none yet" : formatMs(lastMs)}</span>
      </p>
    </Card>
  );
}
