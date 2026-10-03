import { CORRUPTIONS, SEVERITIES } from "../constants.js";
import Card from "./Card.jsx";
import UploadZone from "./UploadZone.jsx";
import SampleGallery from "./SampleGallery.jsx";
import { PrimaryButton } from "./Buttons.jsx";

// The "Input" card shared by the three restoration workspaces:
// upload zone, sample gallery, corruption type (2x2), severity, optional seed, Restore button.
// `workspace` is the object returned by useRestoreWorkspace.
export default function InputPanel({ workspace, samples, backendOnline }) {
  const { image, corruption, setCorruption, severity, setSeverity, seed, setSeed, loading, run } = workspace;
  const petSamples = samples.filter((sample) => sample.kind === "pet");
  const noCorruption = corruption === "none";

  return (
    <Card title="Input" className="w-full lg:w-[340px] lg:shrink-0">
      <div className="flex flex-col gap-5">
        <UploadZone
          selected={image.source?.file ? image.source : null}
          error={image.uploadError}
          onFile={image.chooseFile}
          disabled={loading}
        />

        <Field label="Or pick a sample">
          <SampleGallery
            samples={petSamples}
            selectedId={image.source?.sampleId}
            onSelect={image.chooseSample}
            disabled={loading}
          />
        </Field>

        <Field label="Corruption type">
          <div className="grid grid-cols-2 gap-2">
            {CORRUPTIONS.map((option) => (
              <OptionButton
                key={option.value}
                selected={corruption === option.value}
                disabled={loading}
                onClick={() => setCorruption(option.value)}
              >
                {option.label}
              </OptionButton>
            ))}
          </div>
        </Field>

        <Field label="Severity">
          <SegmentedControl
            options={SEVERITIES}
            value={severity}
            onChange={setSeverity}
            disabled={loading || noCorruption}
          />
        </Field>

        <Field label="Seed (optional)">
          <input
            type="number"
            min="0"
            step="1"
            value={seed}
            placeholder="random"
            disabled={loading || noCorruption}
            onChange={(event) => setSeed(event.target.value)}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 font-mono text-sm disabled:bg-slate-50 disabled:text-slate-400"
          />
        </Field>

        <PrimaryButton
          onClick={run}
          loading={loading}
          loadingText="Restoring..."
          disabled={!image.source || !backendOnline}
        >
          Restore
        </PrimaryButton>
      </div>
    </Card>
  );
}

// A small label above a control.
export function Field({ label, children }) {
  return (
    <div>
      <p className="mb-2 text-xs font-medium text-slate-500">{label}</p>
      {children}
    </div>
  );
}

// One of the 2x2 corruption buttons.
function OptionButton({ selected, disabled, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={`rounded-lg border px-3 py-2 text-left text-sm disabled:cursor-not-allowed disabled:opacity-60 ${
        selected
          ? "border-indigo-600 bg-indigo-50 font-medium text-indigo-700"
          : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
      }`}
    >
      {children}
    </button>
  );
}

// Low / Medium / High segmented control.
export function SegmentedControl({ options, value, onChange, disabled }) {
  return (
    <div className={`flex rounded-lg border border-slate-200 bg-slate-100 p-1 ${disabled ? "opacity-50" : ""}`}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm disabled:cursor-not-allowed ${
            value === option.value ? "bg-white font-medium text-slate-900 shadow-sm" : "text-slate-600"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
