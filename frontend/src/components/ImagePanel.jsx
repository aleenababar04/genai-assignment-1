import Spinner from "./Spinner.jsx";
import Icon from "./Icon.jsx";

// A 288px square image panel (Stitch "image inspection panel"): a header row with the
// label and an optional badge, the image with small overlay tags, and a caption below.
// States: image (src), loading (skeleton + spinner), empty (dashed placeholder).
// `pixelated` keeps sharp pixel edges when a 128x128 model image is enlarged.
// `highlight` gives the output panel its indigo border.
// `fluid` makes the panel fill its column (square, any width) instead of a fixed 288px,
// so three panels fit side by side.
export default function ImagePanel({
  label,
  labelIcon,
  src,
  loading,
  pixelated = true,
  caption,
  badge,
  tag,
  highlight = false,
  fluid = false,
  emptyIcon = "image",
  emptyTitle = "No image yet",
  emptyText,
}) {
  const width = fluid ? "w-full" : "w-72";
  const box = fluid ? "w-full aspect-square" : "size-72";
  let content;
  if (loading) {
    content = (
      <div className="flex size-full animate-pulse flex-col items-center justify-center gap-2 bg-slate-100">
        <Spinner className="size-6 text-primary" />
        <span className="font-mono text-mono-md text-muted">Running model...</span>
      </div>
    );
  } else if (src) {
    content = (
      <>
        <img src={src} alt={label} className={`size-full object-contain ${pixelated ? "pixelated" : ""}`} />
        {pixelated && (
          <span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-0.5 font-mono text-mono-sm text-white backdrop-blur-xs">
            128 × 128 px
          </span>
        )}
        {tag}
      </>
    );
  } else {
    content = (
      <div className="flex size-full flex-col items-center justify-center gap-2 p-6 text-center">
        <span className="mb-1 flex size-12 items-center justify-center rounded-full bg-slate-100 text-slate-400">
          <Icon name={emptyIcon} size={24} />
        </span>
        <p className="text-body-sm font-medium text-slate-600">{emptyTitle}</p>
        {emptyText && <p className="font-mono text-mono-md text-slate-400">{emptyText}</p>}
      </div>
    );
  }

  let frame = "border-2 border-dashed border-line-strong bg-white/50";
  if (src || loading) frame = highlight && src ? "border-2 border-primary bg-white shadow-sm" : "border border-line-strong bg-white shadow-sm";

  return (
    <figure className={`flex flex-col items-center gap-2.5 ${fluid ? "w-full min-w-0" : ""}`}>
      {/* In fluid mode the panels are narrow, so the badge goes on its own line under the
          label (clipped if still too wide) and every header has the same height, which keeps
          the three images aligned. */}
      <figcaption
        className={`flex ${width} px-1 ${fluid ? "min-h-12 flex-col items-start gap-1" : "items-center justify-between gap-2"}`}
      >
        <span className={`flex items-center gap-1 text-body-sm font-semibold ${highlight && src ? "text-primary" : "text-ink"}`}>
          {labelIcon && <Icon name={labelIcon} size={15} className={highlight && src ? "text-primary" : "text-muted"} />}
          {label}
        </span>
        {fluid ? badge && <span className="max-w-full overflow-hidden">{badge}</span> : badge}
      </figcaption>
      <div className={`relative ${box} overflow-hidden rounded-xl ${frame}`}>{content}</div>
      {caption && <p className={`${width} px-1 text-center font-mono text-mono-sm text-muted`}>{caption}</p>}
    </figure>
  );
}

// Small coloured label used in the panel header row and as overlay tags.
export function PanelBadge({ children, className = "border-line bg-slate-100 text-muted" }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-mono text-mono-sm font-medium whitespace-nowrap ${className}`}>
      {children}
    </span>
  );
}

// The arrow between the input and output panels, with a short caption ("Restore").
export function FlowArrow({ label, active = true }) {
  return (
    <div className={`flex flex-col items-center justify-center gap-1 ${active ? "text-primary" : "text-slate-400"}`}>
      <span
        className={`flex size-9 items-center justify-center rounded-full border shadow-xs ${
          active ? "border-primary/30 bg-indigo-50" : "border-line-strong bg-slate-100"
        }`}
      >
        <Icon name="arrow_forward" size={20} />
      </span>
      <span className="font-mono text-mono-sm font-semibold tracking-wide uppercase">{label}</span>
    </div>
  );
}
