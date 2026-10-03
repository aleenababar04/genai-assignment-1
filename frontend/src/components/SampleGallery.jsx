import { apiUrl } from "../api.js";

// "Or pick a sample": a row of small square thumbnails from GET /api/samples.
// `samples` is already filtered to the right kind ("pet" or "face") by the page.
export default function SampleGallery({ samples, selectedId, onSelect, disabled }) {
  if (samples.length === 0) {
    return <p className="text-xs text-slate-500">No sample images available.</p>;
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
            className={`aspect-square overflow-hidden rounded-md border-2 disabled:cursor-not-allowed disabled:opacity-60 ${
              selected ? "border-indigo-600" : "border-transparent hover:border-slate-300"
            }`}
          >
            <img src={apiUrl(sample.url)} alt={sample.name} className="size-full object-cover" />
          </button>
        );
      })}
    </div>
  );
}
