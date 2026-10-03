import { apiUrl } from "../api.js";

// "Or pick a sample": a row of small square thumbnails from GET /api/samples.
// `samples` is already filtered to the right kind ("pet" or "face") by the page.
export default function SampleGallery({ samples, selectedId, onSelect, disabled }) {
  if (samples.length === 0) {
    return <p className="font-mono text-mono-sm text-muted">No sample images available.</p>;
  }
  return (
    <div className="grid grid-cols-6 gap-2">
      {samples.map((sample) => {
        const selected = sample.id === selectedId;
        return (
          <button
            key={sample.id}
            type="button"
            title={sample.name}
            disabled={disabled}
            onClick={() => onSelect(sample)}
            className={`relative aspect-square overflow-hidden rounded-lg transition-all focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 ${
              selected
                ? "border-2 border-primary shadow-xs ring-2 ring-primary/20"
                : "border border-line hover:border-line-strong"
            }`}
          >
            <img src={apiUrl(sample.url)} alt={sample.name} className="size-full object-cover" />
            {selected && <span className="absolute inset-0 bg-primary/10" />}
          </button>
        );
      })}
    </div>
  );
}
