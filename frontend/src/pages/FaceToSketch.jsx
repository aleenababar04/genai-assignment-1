import { useState } from "react";
import { generateSketch } from "../api.js";
import { REQUIRED_MODELS } from "../constants.js";
import useImageSource from "../hooks/useImageSource.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import Card from "../components/Card.jsx";
import { Field } from "../components/InputPanel.jsx";
import UploadZone from "../components/UploadZone.jsx";
import SampleGallery from "../components/SampleGallery.jsx";
import WebcamCapture from "../components/WebcamCapture.jsx";
import StyleSelector from "../components/StyleSelector.jsx";
import { PrimaryButton } from "../components/Buttons.jsx";
import ImagePanel from "../components/ImagePanel.jsx";
import StatTile from "../components/StatTile.jsx";
import DownloadButton from "../components/DownloadButton.jsx";
import ErrorCard from "../components/ErrorCard.jsx";

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

  return (
    <div>
      <PageHeader title="Face-to-Sketch Generator" subtitle="A conditional GAN turns a face photo into a sketch." />
      <ModelNotice health={health} required={REQUIRED_MODELS["face-to-sketch"]} />

      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        {/* Input card */}
        <Card title="Input" className="w-full lg:w-[340px] lg:shrink-0">
          <div className="flex flex-col gap-5">
            <Tabs value={tab} onChange={setTab} disabled={loading} />

            {tab === "upload" ? (
              <>
                <UploadZone
                  selected={image.source?.file ? image.source : null}
                  error={image.uploadError}
                  onFile={image.chooseFile}
                  disabled={loading}
                  prompt="Drop a face photo here or"
                />
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
                {image.source && (
                  <div className="flex items-center gap-3 rounded-lg border border-slate-200 p-2">
                    <img src={image.source.previewUrl} alt="" className="size-12 rounded object-cover" />
                    <div className="text-sm">
                      <p className="text-slate-500">Selected photo</p>
                      <p className="font-mono text-xs text-slate-700">{image.source.name}</p>
                    </div>
                  </div>
                )}
              </>
            )}

            <Field label="Sketch style">
              <StyleSelector value={style} onChange={setStyle} disabled={loading} />
            </Field>

            <PrimaryButton
              onClick={generate}
              loading={loading}
              loadingText="Generating..."
              disabled={!image.source || health.status === "offline"}
            >
              Generate sketch
            </PrimaryButton>
          </div>
        </Card>

        {/* Result card */}
        <div className="min-w-0 flex-1">
          <Card title="Result">
            {error && (
              <div className="mb-4">
                <ErrorCard title="The model could not run" message={error} onDismiss={() => setError(null)} />
              </div>
            )}

            <div className="flex flex-wrap gap-6">
              {result || !loading ? (
                <ImagePanel label="Original photograph" src={result?.input_image} />
              ) : (
                <ImagePanel label="Original photograph" src={image.source?.previewUrl} pixelated={false} caption="Selected photo" />
              )}
              <ImagePanel label="Generated sketch" src={result?.output_image} loading={loading} />
            </div>

            {!result && !loading && (
              <p className="mt-4 text-center text-sm text-slate-500">
                Upload or capture a photo and choose a style to generate a sketch.
              </p>
            )}

            <div className="mt-6 flex flex-col gap-4">
              {(result || loading) && (
                <div>
                  <span className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
                    Style {result?.style ?? style}
                  </span>
                </div>
              )}
              <div className="flex flex-wrap items-end justify-between gap-4">
                <StatTile ms={result?.inference_ms} loading={loading} />
                <DownloadButton
                  dataUrl={result?.output_image}
                  filename={`sketch_style${result?.style ?? style}.png`}
                  label="Download sketch"
                />
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

// Two-tab switch: "Upload photo" / "Use webcam".
function Tabs({ value, onChange, disabled }) {
  const tabs = [
    { value: "upload", label: "Upload photo" },
    { value: "webcam", label: "Use webcam" },
  ];
  return (
    <div className="flex rounded-lg border border-slate-200 bg-slate-100 p-1">
      {tabs.map((tab) => (
        <button
          key={tab.value}
          type="button"
          disabled={disabled}
          aria-pressed={value === tab.value}
          onClick={() => onChange(tab.value)}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm disabled:cursor-not-allowed ${
            value === tab.value ? "bg-white font-medium text-slate-900 shadow-sm" : "text-slate-600"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
