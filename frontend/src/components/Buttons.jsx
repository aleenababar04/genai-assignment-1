import Spinner from "./Spinner.jsx";

// Full-width indigo action button ("Restore", "Generate sketch").
// While `loading` it is disabled and shows a spinner with `loadingText`.
export function PrimaryButton({ children, loading, loadingText, disabled, onClick, className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={`flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500 ${className}`}
    >
      {loading && <Spinner className="size-4 text-slate-500" />}
      {loading ? loadingText : children}
    </button>
  );
}

// White button with a border.
export function SecondaryButton({ children, disabled, onClick, className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400 disabled:hover:bg-white ${className}`}
    >
      {children}
    </button>
  );
}
