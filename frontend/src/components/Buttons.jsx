import Spinner from "./Spinner.jsx";
import Icon from "./Icon.jsx";

// Full-width indigo action button ("Restore Tensor", "Generate sketch").
// While `loading` it is disabled and shows a spinner with `loadingText`.
// `size` "sm" is the 36px button used under the webcam preview.
export function PrimaryButton({ children, icon, loading, loadingText, disabled, onClick, size = "md", className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={`flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 ${size === "sm" ? "h-9 text-body-sm" : "h-10 text-body-md"} font-semibold text-white shadow-sm shadow-indigo-600/20 transition-all hover:bg-primary-hover active:translate-y-0.5 disabled:cursor-not-allowed disabled:border disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none disabled:active:translate-y-0 ${className}`}
    >
      {loading ? <Spinner className="size-4 text-slate-500" /> : icon && <Icon name={icon} size={18} />}
      {loading ? loadingText : children}
    </button>
  );
}

// White button with a border.
export function SecondaryButton({ children, icon, disabled, onClick, className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg border border-line bg-white px-3.5 py-2 text-body-sm font-semibold text-ink shadow-xs transition-colors hover:bg-page disabled:cursor-not-allowed disabled:text-slate-400 disabled:hover:bg-white ${className}`}
    >
      {icon && <Icon name={icon} size={16} />}
      {children}
    </button>
  );
}
