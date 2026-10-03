// Downloads a "data:image/png;base64,..." URL as a PNG file.
// A link with the `download` attribute saves the data URL instead of opening it.
export default function DownloadButton({ dataUrl, filename, label = "Download result" }) {
  const classes = "inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium";

  if (!dataUrl) {
    return (
      <button type="button" disabled className={`${classes} cursor-not-allowed border-slate-200 bg-white text-slate-400`}>
        <DownloadIcon /> {label}
      </button>
    );
  }
  return (
    <a href={dataUrl} download={filename} className={`${classes} border-slate-200 bg-white text-slate-700 hover:bg-slate-50`}>
      <DownloadIcon /> {label}
    </a>
  );
}

function DownloadIcon() {
  return (
    <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 4v11m0 0-4-4m4 4 4-4M5 19h14" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
