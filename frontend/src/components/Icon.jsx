// A Material Symbols Outlined icon (the icon font used in the Stitch screens).
// `name` is the symbol name, e.g. "auto_fix_high"; `size` is the font size in px.
export default function Icon({ name, size = 18, filled = false, className = "" }) {
  return (
    <span
      aria-hidden="true"
      className={`material-symbols-outlined shrink-0 ${filled ? "filled" : ""} ${className}`}
      style={{ fontSize: size }}
    >
      {name}
    </span>
  );
}
