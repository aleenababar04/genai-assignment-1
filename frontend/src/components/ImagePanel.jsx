import Spinner from "./Spinner.jsx";

// A 288px square image panel with a label above and a caption below.
// States: image (src), loading (skeleton + spinner), empty (dashed placeholder).
// `pixelated` keeps sharp pixel edges when a 128x128 model image is enlarged.
export default function ImagePanel({ label, src, loading, pixelated = true, caption = "128 x 128" }) {
  let content;
  if (loading) {
    content = (
      <div className="flex size-full animate-pulse flex-col items-center justify-center gap-2 bg-slate-100">
        <Spinner className="size-6 text-indigo-600" />
        <span className="text-sm text-slate-500">Running model...</span>
      </div>
    );
  } else if (src) {
    content = (
      <img
        src={src}
        alt={label}
        className={`size-full object-contain ${pixelated ? "pixelated" : ""}`}
      />
    );
  } else {
    content = <div className="size-full bg-slate-50" />;
  }

  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="text-sm font-medium text-slate-700">{label}</figcaption>
      <div
        className={`size-72 overflow-hidden rounded-lg border ${
          src || loading ? "border-slate-200" : "border-dashed border-slate-300"
        }`}
      >
        {content}
      </div>
      <p className="text-center font-mono text-xs text-slate-500">{caption}</p>
    </figure>
  );
}
