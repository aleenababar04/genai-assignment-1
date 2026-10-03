import { PAGES, REQUIRED_MODELS } from "../constants.js";
import SystemStatus from "./SystemStatus.jsx";
import Icon from "./Icon.jsx";

// Left sidebar (264px): "Engine workspaces" links to the five pages, the model files
// of the current workspace, and the System status card pinned at the bottom.
// Links change the URL hash; App.jsx listens for that and shows the page.
export default function Sidebar({ currentPage, health }) {
  return (
    <aside className="flex flex-col justify-between gap-6 border-b border-line bg-white lg:sticky lg:top-14 lg:h-[calc(100vh-3.5rem)] lg:w-[264px] lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0">
      <div className="flex flex-col gap-6 p-4">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2 px-2 py-1">
            <Icon name="view_in_ar" size={18} className="text-primary" />
            <span className="font-mono text-mono-md font-semibold tracking-wider text-ink uppercase">Engine Workspaces</span>
          </div>
          <nav className="flex flex-col gap-1">
            {PAGES.map((page) => {
              const active = page.id === currentPage;
              return (
                <a
                  key={page.id}
                  href={`#/${page.id}`}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-body-md transition-colors ${
                    active
                      ? "border-l-2 border-primary bg-primary-soft font-semibold text-primary shadow-xs"
                      : "text-muted hover:bg-page hover:text-ink"
                  }`}
                >
                  <Icon name={page.icon} size={18} filled={active} />
                  <span className="flex-1 leading-tight">{page.title}</span>
                  {active && page.id !== "overview" && <span className="size-1.5 shrink-0 rounded-full bg-primary" />}
                </a>
              );
            })}
          </nav>
        </div>

        {REQUIRED_MODELS[currentPage] && <ModelFiles files={REQUIRED_MODELS[currentPage]} health={health} />}
      </div>

      <div className="border-t border-line p-4">
        <SystemStatus health={health} />
      </div>
    </aside>
  );
}

// "Model topology" box of the Stitch screens, filled with real data: the ONNX files
// this workspace needs and whether the backend has loaded each one.
function ModelFiles({ files, health }) {
  const models = health.data?.models;
  return (
    <div className="flex flex-col gap-2 border-t border-line px-2 pt-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-mono-sm font-semibold tracking-wider text-muted uppercase">Model topology</span>
        <Icon name="device_hub" size={14} className="text-slate-400" />
      </div>
      <div className="flex flex-col gap-1.5 rounded-lg border border-line bg-page p-2.5 font-mono text-mono-sm">
        <div className="flex items-center justify-between gap-2">
          <span className="text-muted">Input:</span>
          <span className="font-semibold text-ink">128 × 128 × 3</span>
        </div>
        {files.map((file) => {
          const loaded = models?.[file]?.loaded;
          return (
            <div key={file} className="flex items-center justify-between gap-2" title={models?.[file]?.error ?? file}>
              <span className="truncate text-muted">{file.replace(/^task\d_/, "").replace(".onnx", "")}</span>
              <span className={`shrink-0 font-semibold ${models ? (loaded ? "text-online" : "text-red-600") : "text-muted"}`}>
                {models ? (loaded ? "Loaded" : "Missing") : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
