// "Sketch style": three selectable cards, Style 1 / Style 2 / Style 3.
// The preview is a small drawn face whose line style differs per style; it is only
// a hint, not real model output. Replace it with real example sketches if you have them.
const STYLES = [
  { value: 1, label: "Style 1", stroke: 1.2, dash: "" },
  { value: 2, label: "Style 2", stroke: 2.4, dash: "" },
  { value: 3, label: "Style 3", stroke: 1.4, dash: "3 2" },
];

export default function StyleSelector({ value, onChange, disabled }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {STYLES.map((style) => {
        const selected = style.value === value;
        return (
          <button
            key={style.value}
            type="button"
            disabled={disabled}
            aria-pressed={selected}
            onClick={() => onChange(style.value)}
            className={`flex flex-col items-center gap-1 rounded-lg p-2 disabled:cursor-not-allowed disabled:opacity-60 ${
              selected ? "border-2 border-indigo-600 bg-indigo-50" : "border-2 border-slate-200 bg-white hover:bg-slate-50"
            }`}
          >
            <FacePreview stroke={style.stroke} dash={style.dash} />
            <span className={`text-sm ${selected ? "font-semibold text-indigo-700" : "text-slate-700"}`}>{style.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function FacePreview({ stroke, dash }) {
  return (
    <svg viewBox="0 0 40 40" className="size-12 rounded bg-white text-slate-700" fill="none" stroke="currentColor"
      strokeWidth={stroke} strokeDasharray={dash} strokeLinecap="round" aria-hidden="true">
      <ellipse cx="20" cy="21" rx="11" ry="14" />
      <path d="M14 18h3M23 18h3M20 20v5M16 29q4 3 8 0" />
    </svg>
  );
}
