import Icon from "./Icon.jsx";

// Downloads a "data:image/png;base64,..." URL as a PNG file.
// A link with the `download` attribute saves the data URL instead of opening it.
// `primary` gives it the filled indigo look (Face-to-Sketch); otherwise it is the white button.
export default function DownloadButton({ dataUrl, filename, label = "Download result", primary = false, className = "" }) {
  const classes = `inline-flex items-center justify-center gap-1.5 rounded-lg border px-4 py-2.5 text-body-sm font-semibold shadow-xs transition-colors ${className}`;

  if (!dataUrl) {
    return (
      <button type="button" disabled className={`${classes} cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400 shadow-none`}>
        <Icon name="download" size={18} /> {label}
      </button>
    );
  }
  const look = primary
    ? "border-primary bg-primary text-white hover:bg-primary-hover"
    : "border-line bg-white text-ink hover:bg-page";
  return (
    <a href={dataUrl} download={filename} className={`${classes} ${look}`}>
      <Icon name="download" size={18} /> {label}
    </a>
  );
}
