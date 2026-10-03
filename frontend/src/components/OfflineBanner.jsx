import { SecondaryButton } from "./Buttons.jsx";

// Shown at the top of the main area when /api/health cannot be reached.
export default function OfflineBanner({ onRetry }) {
  return (
    <div role="alert" className="mb-6 flex items-center gap-4 rounded-xl border border-red-300 bg-red-50 px-5 py-3">
      <span className="size-2.5 shrink-0 rounded-full bg-red-600" />
      <p className="flex-1 text-sm text-red-700">
        Cannot reach the backend. Check that the server is running and try again.
      </p>
      <SecondaryButton onClick={onRetry}>Retry</SecondaryButton>
    </div>
  );
}
