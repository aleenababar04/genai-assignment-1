import { CATEGORIES, CORRUPTIONS, SEVERITIES } from "../constants.js";
import Card, { SectionLabel } from "./Card.jsx";
import UploadZone from "./UploadZone.jsx";
import SampleGallery from "./SampleGallery.jsx";
import { PrimaryButton } from "./Buttons.jsx";
import Icon from "./Icon.jsx";

// The "Input Configuration" card shared by the three restoration workspaces:
// upload zone, sample gallery, corruption type (2x2), severity, optional seed, Restore button.
// `workspace` is the object returned by useRestoreWorkspace.
// `buttonLabel` / `buttonIcon` give each workspace its own action wording (from the Stitch screens).
export default function InputPanel({ workspace, samples, backendOnline, buttonLabel = "Restore", buttonIcon = "auto_fix_high" }) {
  const { image, corruption, setCorruption, severity, setSeverity, seed, setSeed, loading, run, result } = workspace;
  const petSamples = samples.filter((sample) => sample.kind === "pet");
  const noCorruption = corruption === "none";
  const hint = severityHint(result?.corruption, corruption, severity);

  return (
    <Card title="Input Configuration" subtitle="Upload an image or pick a sample" icon="input">
      <UploadZone
        selected={image.source?.file ? image.source : null}
        error={image.uploadError}
        onFile={image.chooseFile}
        disabled={loading}
      />

      <Field label="Or pick a sample" aside={petSamples.length > 0 ? `${petSamples.length} samples · 128px` : null}>
        <SampleGallery
          samples={petSamples}
          selectedId={image.source?.sampleId}
          onSelect={image.chooseSample}
          disabled={loading}
        />
      </Field>

      <div className="border-t border-line pt-4">
        <Field label="Corruption type">
          <div className="grid grid-cols-2 gap-2">
            {CORRUPTIONS.map((option) => (
              <OptionButton
                key={option.value}
                option={option}
                selected={corruption === option.value}
                disabled={loading}
                onClick={() => setCorruption(option.value)}
              />
            ))}
          </div>
        </Field>
      </div>

      <Field label="Severity" aside={hint} asideClass="text-primary">
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
          className="h-9 w-full rounded-lg border border-line bg-white px-3 font-mono text-mono-lg text-ink placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/20 focus:outline-none disabled:bg-page disabled:text-slate-400"
        />
      </Field>

      <PrimaryButton
        icon={buttonIcon}
        onClick={run}
        loading={loading}
        loadingText="Restoring..."
        disabled={!image.source || !backendOnline}
        className="mt-1"
      >
        {buttonLabel}
      </PrimaryButton>
    </Card>
  );
}

// The parameter the backend used in the last run ("p = 0.08"), shown next to "Severity"
// while the current selection still matches that run. Nothing is shown before a run.
function severityHint(settings, corruption, severity) {
  if (!settings || settings.type !== corruption || settings.severity !== severity) return null;
  if (settings.probability !== undefined) return `p = ${settings.probability} intensity`;
  if (settings.sigma !== undefined) return `σ = ${settings.sigma}, k = ${settings.kernel_size}`;
  if (settings.target_coverage !== undefined) return `coverage = ${Math.round(settings.target_coverage * 100)}%`;
  return null;
}

// A section label above a control, with optional mono text on the right.
export function Field({ label, aside, asideClass = "text-muted", children }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <SectionLabel>{label}</SectionLabel>
        {aside && <span className={`font-mono text-mono-sm font-medium ${asideClass}`}>{aside}</span>}
      </div>
      {children}
    </div>
  );
}

// One of the 2x2 corruption option cards: colour dot, name and a short hint.
function OptionButton({ option, selected, disabled, onClick }) {
  const category = CATEGORIES[option.category];
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={`relative flex items-center gap-2 rounded-lg p-2.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
        selected ? "border-2 border-primary bg-primary-soft shadow-xs" : "border border-line bg-white hover:bg-page"
      }`}
    >
      <span className={`size-2.5 shrink-0 rounded-full ${category.bar}`} />
      <span className="flex min-w-0 flex-col">
        <span className={`text-body-sm leading-tight font-semibold ${selected ? "text-primary" : "text-ink"}`}>{option.short}</span>
        <span className={`font-mono text-mono-sm ${selected ? "text-primary/80" : "text-muted"}`}>{option.hint}</span>
      </span>
      {selected && <Icon name="check_circle" size={16} filled className="absolute top-1.5 right-1.5 text-primary" />}
    </button>
  );
}

// Low / Medium / High segmented control.
export function SegmentedControl({ options, value, onChange, disabled }) {
  return (
    <div className={`grid grid-flow-col gap-1 rounded-lg border border-line bg-slate-100 p-1 ${disabled ? "opacity-50" : ""}`}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 text-body-sm transition-colors disabled:cursor-not-allowed ${
            value === option.value ? "bg-white font-semibold text-ink shadow-xs" : "font-medium text-muted hover:text-ink"
          }`}
        >
          {option.icon && <span className="material-symbols-outlined" style={{ fontSize: 16 }} aria-hidden="true">{option.icon}</span>}
          {option.label}
        </button>
      ))}
    </div>
  );
}
