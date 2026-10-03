import { useState } from "react";
import { generateSketch } from "../api.js";
import { REQUIRED_MODELS, formatMs } from "../constants.js";
import useImageSource from "../hooks/useImageSource.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import Card from "../components/Card.jsx";
import { Field, SegmentedControl } from "../components/InputPanel.jsx";
import UploadZone from "../components/UploadZone.jsx";
import SampleGallery from "../components/SampleGallery.jsx";
import WebcamCapture from "../components/WebcamCapture.jsx";
import StyleSelector from "../components/StyleSelector.jsx";
import { PrimaryButton } from "../components/Buttons.jsx";
import ImagePanel, { FlowArrow, PanelBadge } from "../components/ImagePanel.jsx";
import StatTile, { InfoTile } from "../components/StatTile.jsx";
import DownloadButton from "../components/DownloadButton.jsx";
import ErrorCard from "../components/ErrorCard.jsx";
import Icon from "../components/Icon.jsx";

const TABS = [
  { value: "upload", label: "Upload photo", icon: "upload_file" },
  { value: "webcam", label: "Use webcam", icon: "photo_camera" },
];

// Workspace 4: a conditional GAN turns a face photo into a sketch in one of three styles.
export default function FaceToSketch({ health, samples, onDone }) {
  const image = useImageSource();
  const [tab, setTab] = useState("upload"); // "upload" or "webcam"
  const [style, setStyle] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const faceSamples = samples.filter((sample) => sample.kind === "face");

  async function generate() {
    if (!image.source) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await generateSketch(image.source, style));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      onDone?.();
    }
  }

  const shownStyle = result?.style ?? style;

  return (
    <div>
      <PageHeader
        title="Face-to-Sketch Generator"
        subtitle="A conditional GAN turns a face photo into a sketch."
        badge="Pipeline 04: Conditional GAN"
        eyebrow="Stage: cross-domain synthesis"
      />
      <ModelNotice health={health} required={REQUIRED_MODELS["face-to-sketch"]} />

      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        {/* Input card */}
        <Card title="Input Acquisition" subtitle="Upload a face photo or use the webcam" icon="input">
          <SegmentedControl options={TABS} value={tab} onChange={setTab} disabled={loading} />

          {tab === "upload" ? (
            <>
              <UploadZone
                selected={null}
                error={image.uploadError}
                onFile={image.chooseFile}
                disabled={loading}
                prompt="Drop a face photo here or"
              />
              {image.source && <SelectedPhoto source={image.source} />}
              {faceSamples.length > 0 && (
                <Field label="Or pick a sample">
                  <SampleGallery
                    samples={faceSamples}
                    selectedId={image.source?.sampleId}
                    onSelect={image.chooseSample}
                    disabled={loading}
                  />
                </Field>
              )}
            </>
          ) : (
            <>
              <WebcamCapture onCapture={image.chooseFile} disabled={loading} />
              {image.source && <SelectedPhoto source={image.source} />}
            </>
          )}

          <Field label="Sketch style" aside="3 styles" asideClass="text-primary">
            <StyleSelector value={style} onChange={setStyle} disabled={loading} />
          </Field>

          <div className="flex flex-col gap-2 pt-1">
            <PrimaryButton
              icon="auto_fix_high"
              onClick={generate}
              loading={loading}
              loadingText="Generating..."
              disabled={!image.source || health.status === "offline"}
            >
              Generate sketch
            </PrimaryButton>
            <p className="text-center font-mono text-mono-sm text-muted">Output: 128×128 sketch</p>
          </div>
        </Card>

        {/* Result card */}
        <Card
          title="Synthesis Inspection"
          subtitle="Original photograph and generated sketch, side by side"
          aside={<PanelBadge className="border-indigo-200 bg-indigo-50 font-semibold text-indigo-700 uppercase">Conditional GAN</PanelBadge>}
        >
          {error && <ErrorCard title="The model could not run" message={error} onDismiss={() => setError(null)} />}

          <div className="flex flex-wrap items-center justify-center gap-4 rounded-xl border border-line bg-page p-4">
            {result || !loading ? (
              <ImagePanel
                label="Original photograph"
                labelIcon="image"
                src={result?.input_image}
                badge={<PanelBadge>128 × 128</PanelBadge>}
                emptyIcon="photo_camera"
                emptyTitle="No input photograph"
                emptyText={tab === "webcam" ? "Awaiting capture from webcam" : "Upload or pick a photo"}
              />
            ) : (
              <ImagePanel label="Original photograph" labelIcon="image" src={image.source?.previewUrl} pixelated={false} caption="Selected photo" />
            )}
            <FlowArrow label="cGAN" active={Boolean(result || loading)} />
            <ImagePanel
              label="Generated sketch"
              labelIcon="draw"
              src={result?.output_image}
              loading={loading}
              highlight
              badge={
                result ? (
                  <PanelBadge className="border-emerald-200 bg-emerald-50 font-semibold text-emerald-700">Ready</PanelBadge>
                ) : (
                  <PanelBadge>Awaiting</PanelBadge>
                )
              }
              tag={
                <span className="absolute bottom-2 right-2 rounded border border-indigo-400/30 bg-indigo-950/85 px-2 py-0.5 font-mono text-mono-sm text-indigo-200">
                  STYLE {result?.style}
                </span>
              }
              emptyIcon="auto_fix_high"
              emptyTitle="No sketch generated"
              emptyText="Output will render at 128×128"
            />
          </div>

          {!result && !loading && (
            <p className="-mt-1 text-center text-body-md text-muted">
              Upload or capture a photo and choose a style to generate a sketch.
            </p>
          )}

          <div className="flex flex-wrap items-center gap-2 border-t border-line pt-4">
            <span className="mr-1 text-label-ui text-muted">Inference attributes:</span>
            <AttrChip icon="palette" active>Style {shownStyle}</AttrChip>
            <AttrChip icon="timer" active={Boolean(result)}>
              Inference time: {result ? formatMs(result.inference_ms) : "—"}
            </AttrChip>
            <AttrChip icon="grid_4x4">Resolution: 128 x 128</AttrChip>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatTile ms={result?.inference_ms} loading={loading} />
            <InfoTile label="Model">{REQUIRED_MODELS["face-to-sketch"][0]}</InfoTile>
            <DownloadButton
              dataUrl={result?.output_image}
              filename={`sketch_style${result?.style ?? style}.png`}
              label="Download sketch"
              primary
              className="h-full"
            />
          </div>
        </Card>
      </div>
    </div>
  );
}

// The picked photo (upload, sample or webcam capture) as a small file card.
function SelectedPhoto({ source }) {
  const kind = source.sampleId ? "Sample" : source.name === "webcam_capture.png" ? "Captured" : "Uploaded";
  return (
    <div className="flex items-center gap-3 rounded-lg border border-line bg-page p-3">
      <img src={source.previewUrl} alt="" className="size-10 shrink-0 rounded border border-line-strong bg-white object-cover shadow-xs" />
      <div className="min-w-0 space-y-0.5">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-body-sm font-semibold text-ink">{source.name}</span>
          <span className="shrink-0 rounded border border-emerald-200 bg-emerald-50 px-1 font-mono text-mono-sm font-medium text-emerald-700">
            {kind}
          </span>
        </div>
        <p className="font-mono text-mono-sm text-muted">
          {source.file ? `${Math.max(1, Math.round(source.file.size / 1024))} KB · ` : ""}resized to 128×128
        </p>
      </div>
    </div>
  );
}

// A chip in the "Inference attributes" row. `active` = indigo (has a value), otherwise grey.
function AttrChip({ icon, active = false, children }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-2.5 py-1 font-mono text-mono-sm ${
        active ? "border-indigo-200 bg-indigo-50 font-semibold text-indigo-700" : "border-line bg-slate-100 text-ink"
      }`}
    >
      <Icon name={icon} size={14} className={active ? "" : "text-muted"} />
      {children}
    </span>
  );
}
