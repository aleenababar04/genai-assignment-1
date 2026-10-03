import { useRef, useState } from "react";

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
  let look = "border-slate-300 bg-slate-50 hover:border-indigo-400";
  if (error) look = "border-red-400 bg-red-50";
  else if (dragging) look = "border-indigo-500 bg-indigo-50";

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
      className={`flex min-h-32 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-4 text-center ${look} ${
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
          <span className="flex size-7 items-center justify-center rounded-full bg-red-600 text-sm font-bold text-white">!</span>
          <p className="font-mono text-xs text-red-700">{error.name}</p>
          <p className="text-sm text-red-700">{error.message}</p>
          <span className="text-sm font-medium text-indigo-600 underline">Choose another file</span>
        </>
      ) : selected ? (
        <>
          <img src={selected.previewUrl} alt="" className="size-14 rounded-md border border-slate-200 object-cover" />
          <p className="max-w-full truncate font-mono text-xs text-slate-700">{selected.name}</p>
          <span className="text-xs text-indigo-600 underline">Choose another file</span>
        </>
      ) : (
        <>
          <UploadIcon />
          <p className="text-sm text-slate-700">
            {prompt} <span className="font-medium text-indigo-600 underline">browse</span>
          </p>
        </>
      )}
      <p className="text-xs text-slate-500">PNG or JPG</p>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg className="size-7 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
