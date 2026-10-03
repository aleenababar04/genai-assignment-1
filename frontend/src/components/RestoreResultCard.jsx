import Card from "./Card.jsx";
import ImagePanel from "./ImagePanel.jsx";
import CorruptionChips from "./CorruptionChips.jsx";
import StatTile from "./StatTile.jsx";
import DownloadButton from "./DownloadButton.jsx";
import ErrorCard from "./ErrorCard.jsx";

// "Result" card of the three restoration workspaces: input and output panels,
// corruption settings chips, inference time and the download button.
// `workspace` is the object returned by useRestoreWorkspace.
export default function RestoreResultCard({ workspace, outputLabel, filename }) {
  const { result, loading, error, clearError, image, corruption, severity } = workspace;

  // While the model runs, show the picked image on the left (not yet corrupted)
  // and the settings that were sent.
  const pendingSettings = corruption === "none" ? { type: "none" } : { type: corruption, severity };
  const settings = result?.corruption ?? (loading ? pendingSettings : null);

  return (
    <Card title="Result">
      {error && (
        <div className="mb-4">
          <ErrorCard title="The model could not run" message={error} onDismiss={clearError} />
        </div>
      )}

      <div className="flex flex-wrap gap-6">
        {result || !loading ? (
          <ImagePanel label="Input image" src={result?.input_image} />
        ) : (
          <ImagePanel label="Input image" src={image.source?.previewUrl} pixelated={false} caption="Selected image" />
        )}
        <ImagePanel label={outputLabel} src={result?.output_image} loading={loading} />
      </div>

      {!result && !loading && (
        <p className="mt-4 text-center text-sm text-slate-500">
          No result yet. Upload an image or pick a sample to begin.
        </p>
      )}

      <div className="mt-6 flex flex-col gap-4">
        <CorruptionChips settings={settings} />
        <div className="flex flex-wrap items-end justify-between gap-4">
          <StatTile ms={result?.inference_ms} loading={loading} />
          <DownloadButton dataUrl={result?.output_image} filename={filename} />
        </div>
      </div>
    </Card>
  );
}
