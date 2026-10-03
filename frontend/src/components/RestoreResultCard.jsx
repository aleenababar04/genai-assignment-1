import { CATEGORIES, CORRUPTIONS } from "../constants.js";
import Card from "./Card.jsx";
import ImagePanel, { FlowArrow, PanelBadge } from "./ImagePanel.jsx";
import CorruptionChips from "./CorruptionChips.jsx";
import StatTile, { InfoTile } from "./StatTile.jsx";
import DownloadButton from "./DownloadButton.jsx";
import ErrorCard from "./ErrorCard.jsx";
import Icon from "./Icon.jsx";

// "Restoration Inspection" card of the three restoration workspaces: input and output
// panels side by side, corruption settings chips, inference time, model file(s) and download.
// `workspace` is the object returned by useRestoreWorkspace.
// `flowLabel` is the caption of the arrow between the panels; `outputBadge(result)` the
// text of the badge above the output panel; `models` the ONNX files the workspace uses.
export default function RestoreResultCard({ workspace, outputLabel, filename, flowLabel = "Restore", outputBadge, models = [] }) {
  const { result, loading, error, clearError, image, corruption, severity } = workspace;

  // While the model runs, show the picked image on the left (not yet corrupted)
  // and the settings that were sent.
  const pendingSettings = corruption === "none" ? { type: "none" } : { type: corruption, severity };
  const settings = result?.corruption ?? (loading ? pendingSettings : null);

  const type = result?.corruption?.type;
  const category = type ? CATEGORIES[type === "none" ? "clean" : type] : null;
  const typeLabel = type === "none" ? "Clean input" : CORRUPTIONS.find((c) => c.value === type)?.short;
  const outText = result && outputBadge ? outputBadge(result) : null;

  return (
    <Card title="Restoration Inspection" subtitle="Side-by-side 128×128 pixel comparison">
      {error && <ErrorCard title="The model could not run" message={error} onDismiss={clearError} />}

      <div className="flex flex-wrap items-center justify-center gap-4 rounded-xl border border-line bg-page p-4">
        {result || !loading ? (
          <ImagePanel
            label="Input image"
            src={result?.input_image}
            badge={category && <PanelBadge className={`${category.soft} ${category.text} ${category.softBorder}`}>{typeLabel}</PanelBadge>}
            tag={<OverlayTag>SRC</OverlayTag>}
            emptyIcon="image"
            emptyTitle="No input image"
            emptyText="Upload or pick a sample"
          />
        ) : (
          <ImagePanel label="Input image" src={image.source?.previewUrl} pixelated={false} caption="Selected image" />
        )}
        <FlowArrow label={flowLabel} active={Boolean(result || loading)} />
        <ImagePanel
          label={outputLabel}
          src={result?.output_image}
          loading={loading}
          highlight
          badge={outText && <PanelBadge className="border-emerald-200 bg-emerald-50 text-emerald-700">{outText}</PanelBadge>}
          tag={
            <span className="absolute top-2 right-2 flex items-center gap-1 rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-mono text-mono-sm font-semibold text-emerald-700">
              <Icon name="check" size={12} /> OUTPUT
            </span>
          }
          emptyIcon="auto_fix_high"
          emptyTitle="No result yet"
          emptyText="Output renders at 128×128"
        />
      </div>

      {!result && !loading && (
        <p className="-mt-1 text-center text-body-md text-muted">No result yet. Upload an image or pick a sample to begin.</p>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-line bg-page px-3 py-2 font-mono text-mono-sm text-muted">
        <span className="flex items-center gap-1 font-medium text-ink">
          <Icon name="view_in_ar" size={15} className="text-primary" />
          128 × 128 shown at 2.25× (288 px)
        </span>
        <span>•</span>
        <span>Interpolation: nearest neighbour</span>
      </div>

      <CorruptionChips settings={settings} />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile ms={result?.inference_ms} loading={loading} />
        <InfoTile label={models.length > 1 ? "Models" : "Model"}>
          {models.map((name) => (
            <span key={name} className="block truncate" title={name}>{name}</span>
          ))}
        </InfoTile>
        <DownloadButton dataUrl={result?.output_image} filename={filename} className="h-full" />
      </div>
    </Card>
  );
}

// Small tag in the top-right corner of an image panel.
function OverlayTag({ children }) {
  return (
    <span className="absolute top-2 right-2 rounded border border-line bg-white/90 px-1.5 py-0.5 font-mono text-mono-sm text-ink">
      {children}
    </span>
  );
}
