import { PAGES, REQUIRED_MODELS, formatMs } from "../constants.js";
import Icon from "../components/Icon.jsx";

// One card per workspace on the Overview page (colours and wording from the Stitch screen).
// The chips are descriptive facts; the model chip shows the real load state from /api/health.
const WORKSPACES = [
  {
    id: "universal",
    description: "One autoencoder restores any corruption.",
    tag: "All corruptions",
    tile: "bg-indigo-50 text-primary",
    tagClass: "border-emerald-200 bg-emerald-50 text-emerald-700",
    chips: ["Input: 128×128", "Single-model baseline"],
    footer: "Pipeline 01 • Denoising autoencoder",
  },
  {
    id: "hard-routed",
    description: "A classifier detects the corruption and routes the image to one specialist.",
    tag: "Classifier directed",
    tile: "bg-amber-100 text-amber-500",
    tagClass: "border-amber-200 bg-amber-50 text-amber-700",
    chips: ["Classifier + 3 specialists", "Discrete dispatch"],
    footer: "Pipeline 02 • Classifier router",
  },
  {
    id: "soft-moe",
    description: "A gating network blends four expert branches.",
    tag: "Gating network",
    tile: "bg-cyan-50 text-cyan-500",
    tagClass: "border-cyan-200 bg-cyan-50 text-cyan-700",
    chips: ["Continuous soft weights", "Weighted-sum output"],
    footer: "Pipeline 03 • Soft MoE (4 branches)",
  },
  {
    id: "face-to-sketch",
    description: "A conditional GAN turns a face photo into a sketch in three styles.",
    tag: "Conditional GAN",
    tile: "bg-fuchsia-50 text-fuchsia-500",
    tagClass: "border-fuchsia-200 bg-fuchsia-50 text-fuchsia-700",
    chips: ["Style 1 / Style 2 / Style 3", "Upload or webcam"],
    footer: "Pipeline 04 • Conditional GAN",
  },
];

export default function Overview({ health }) {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col justify-between gap-4 border-b border-line pb-4 md:flex-row md:items-end">
        <div>
          <h1 className="text-headline-xl text-ink">Overview</h1>
          <p className="mt-1 text-body-lg text-muted">
            Four workspaces for restoring corrupted images and generating sketches from face photos.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="inline-flex items-center rounded border border-line bg-white px-2.5 py-1 font-mono text-mono-md text-ink shadow-sm">
            <span className="mr-1.5 size-1.5 rounded-full bg-primary" />
            Resolution: 128×128
          </span>
          <span className="inline-flex items-center rounded border border-line bg-white px-2.5 py-1 font-mono text-mono-md text-ink shadow-sm">
            <Icon name="bolt" size={14} className="mr-1 text-muted" />
            ONNX Runtime
          </span>
        </div>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        {WORKSPACES.map((workspace, index) => (
          <WorkspaceCard key={workspace.id} workspace={workspace} primary={index === 0} health={health} />
        ))}
      </div>

      <SystemCard health={health} />
    </div>
  );
}

function WorkspaceCard({ workspace, primary, health }) {
  const page = PAGES.find((p) => p.id === workspace.id);
  const required = REQUIRED_MODELS[workspace.id];
  const models = health.data?.models;
  const loaded = models ? required.filter((name) => models[name]?.loaded).length : null;

  return (
    <div className="flex flex-col justify-between rounded-xl border border-line bg-white p-6 shadow-sm transition-shadow hover:shadow-md">
      <div>
        <div className="flex items-start justify-between">
          <span className={`flex size-10 items-center justify-center rounded-lg ${workspace.tile}`}>
            <Icon name={page.icon} size={22} />
          </span>
          <span className={`inline-flex h-5 items-center rounded border px-2 font-mono text-mono-sm font-medium uppercase ${workspace.tagClass}`}>
            {workspace.tag}
          </span>
        </div>
        <div className="mt-4">
          <h2 className="text-headline-md text-ink">{page.title}</h2>
          <p className="mt-1 text-body-md text-muted">{workspace.description}</p>
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {workspace.chips.map((chip) => (
            <span key={chip} className="rounded border border-line bg-page px-2 py-1 font-mono text-mono-sm text-slate-600">
              {chip}
            </span>
          ))}
          {loaded !== null && (
            <span
              className={`rounded border px-2 py-1 font-mono text-mono-sm ${
                loaded === required.length
                  ? "border-green-200 bg-green-50 text-green-700"
                  : "border-red-200 bg-red-50 text-red-700"
              }`}
            >
              Models: {loaded} of {required.length} loaded
            </span>
          )}
        </div>
      </div>
      <div className="mt-6 flex items-center justify-between gap-3 border-t border-line pt-4">
        <span className="text-label-ui text-slate-400">{workspace.footer}</span>
        <a
          href={`#/${workspace.id}`}
          className={`flex h-9 shrink-0 items-center gap-1.5 rounded px-4 text-label-ui font-medium transition-all active:translate-y-0.5 ${
            primary
              ? "bg-primary text-white hover:bg-primary-hover"
              : "border border-line bg-white text-ink hover:border-line-strong hover:bg-page"
          }`}
        >
          Open workspace
          <Icon name={primary ? "arrow_forward" : "chevron_right"} size={16} />
        </a>
      </div>
    </div>
  );
}

// Full-width "System" card: backend status, one module tile per ONNX model, last inference time.
function SystemCard({ health }) {
  const models = health.data?.models ?? {};
  const lastMs = health.data?.last_inference_ms;
  const online = health.status === "online";

  return (
    <section className="rounded-xl border border-line bg-white p-6 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-line pb-3">
        <div className="flex items-center gap-2">
          <Icon name="memory" size={22} className="text-primary" />
          <h2 className="text-headline-sm text-ink">System Status</h2>
        </div>
        <div className="flex items-center gap-2 font-mono text-mono-sm text-muted">
          <span className="relative flex size-2">
            {online && <span className="absolute inline-flex size-full animate-ping rounded-full bg-online opacity-75" />}
            <span className={`relative inline-flex size-2 rounded-full ${online ? "bg-online" : health.status === "offline" ? "bg-red-600" : "bg-slate-400"}`} />
          </span>
          <span className="font-medium text-ink">
            {online ? "Backend online" : health.status === "offline" ? "Backend offline" : "Checking backend..."}
          </span>
          <span>•</span>
          <span>REST /api</span>
        </div>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-12">
        <div className="space-y-2 lg:col-span-8">
          <span className="block text-label-ui tracking-wider text-slate-400 uppercase">ONNX models</span>
          {Object.keys(models).length === 0 ? (
            <p className="text-body-md text-muted">No model information (backend not reachable).</p>
          ) : (
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
              {Object.entries(models).map(([name, info]) => (
                <div key={name} className="flex flex-col gap-1.5 rounded-lg border border-line bg-page p-2.5" title={info.error ?? "Loaded"}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-mono text-mono-md font-semibold text-ink">{name}</span>
                    <span
                      className={`shrink-0 rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium ${
                        info.loaded ? "border-green-200 bg-green-100 text-online" : "border-red-200 bg-red-50 text-red-600"
                      }`}
                    >
                      {info.loaded ? "Loaded" : "Not loaded"}
                    </span>
                  </div>
                  <span className="truncate font-mono text-mono-sm text-slate-400">{info.loaded ? "Ready for inference" : info.error ?? "File not found"}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-2 rounded-lg border border-line bg-page p-3 lg:col-span-4">
          <span className="block text-label-ui tracking-wider text-slate-400 uppercase">Instrumentation metrics</span>
          <div className="space-y-1.5 text-body-sm">
            <Metric label="Last inference:" value={lastMs == null ? "none yet" : formatMs(lastMs)} accent={lastMs != null} />
            <Metric
              label="ONNX models:"
              value={health.data ? `${health.data.loaded_count} of ${health.data.expected_count} loaded` : "—"}
            />
            <Metric label="Input matrix:" value="128×128 RGB" />
          </div>
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value, accent = false }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-muted">{label}</span>
      <span className={`font-mono text-mono-md font-semibold ${accent ? "text-primary" : "text-ink"}`}>{value}</span>
    </div>
  );
}
