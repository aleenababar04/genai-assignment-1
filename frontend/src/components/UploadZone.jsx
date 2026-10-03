import { useRef, useState } from "react";
import Icon from "./Icon.jsx";

// Dashed drag-and-drop zone. Clicking it opens the file browser.
// Props:
//   selected     {name, previewUrl} of the chosen file, shown with a thumbnail (or null)
//   error        {name, message} of a rejected file, shown in red (or null)
//   onFile(file) called with the dropped/selected File (the type check happens in useImageSource)
export default function UploadZone({ selected, error, onFile, disabled, prompt = "Drop an image here or" }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function openBrowser() {
    if (!disabled) inputRef.current?.click();
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  }

  function handleChange(event) {
    const file = event.target.files?.[0];
    if (file) onFile(file);
    event.target.value = ""; // allow choosing the same file again
  }

  // Colours for the three looks: error (red), dragging (indigo), normal (grey).
  let look = "border-line-strong bg-page hover:border-primary";
  if (error) look = "border-red-300 bg-red-50";
  else if (dragging) look = "border-primary bg-primary-soft";

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={openBrowser}
      onKeyDown={(event) => (event.key === "Enter" || event.key === " ") && openBrowser()}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`group flex min-h-36 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-5 text-center transition-all ${look} ${
        disabled ? "cursor-not-allowed opacity-60" : ""
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />

      {error ? (
        <>
          <span className="flex size-10 items-center justify-center rounded-full border border-red-200 bg-white text-red-600 shadow-xs">
            <Icon name="error" size={20} />
          </span>
          <p className="max-w-full truncate font-mono text-mono-md text-red-700">{error.name}</p>
          <p className="text-body-sm text-red-700">{error.message}</p>
          <span className="text-body-sm font-semibold text-primary underline">Choose another file</span>
        </>
      ) : selected ? (
        <>
          <img src={selected.previewUrl} alt="" className="size-14 rounded-lg border border-line object-cover shadow-xs" />
          <p className="max-w-full truncate font-mono text-mono-md text-ink">{selected.name}</p>
          <span className="text-body-sm font-semibold text-primary underline">Choose another file</span>
        </>
      ) : (
        <>
          <span className="flex size-10 items-center justify-center rounded-full border border-line bg-white text-primary shadow-xs transition-transform group-hover:scale-105">
            <Icon name="cloud_upload" size={20} />
          </span>
          <p className="text-body-sm font-semibold text-ink transition-colors group-hover:text-primary">
            {prompt} <span className="text-primary underline">browse</span>
          </p>
        </>
      )}
      <p className="font-mono text-mono-sm text-muted">PNG or JPG (resized to 128x128)</p>
    </div>
  );
}
