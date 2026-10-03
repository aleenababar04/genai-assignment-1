import Icon from "./Icon.jsx";

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
            className={`group relative flex flex-col items-center rounded-lg p-2 transition-all disabled:cursor-not-allowed disabled:opacity-60 ${
              selected
                ? "border-2 border-primary bg-indigo-50/30 shadow-xs"
                : "border border-line bg-white hover:border-line-strong hover:shadow-xs"
            }`}
          >
            {selected && (
              <span className="absolute -top-1.5 -right-1.5 flex size-4 items-center justify-center rounded-full bg-primary text-white shadow-xs">
                <Icon name="check" size={12} />
              </span>
            )}
            <FacePreview stroke={style.stroke} dash={style.dash} selected={selected} />
            <span className={`text-label-ui ${selected ? "font-bold text-primary" : "font-medium text-ink group-hover:text-primary"}`}>
              {style.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function FacePreview({ stroke, dash, selected }) {
  return (
    <div className={`mb-1.5 aspect-square w-full overflow-hidden rounded border bg-slate-50 ${selected ? "border-indigo-200" : "border-line"}`}>
      <svg viewBox="0 0 40 40" className="size-full text-slate-700" fill="none" stroke="currentColor"
        strokeWidth={stroke} strokeDasharray={dash} strokeLinecap="round" aria-hidden="true">
        <ellipse cx="20" cy="21" rx="11" ry="14" />
        <path d="M14 18h3M23 18h3M20 20v5M16 29q4 3 8 0" />
      </svg>
    </div>
  );
}
