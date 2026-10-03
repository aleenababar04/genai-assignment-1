import { SecondaryButton } from "./Buttons.jsx";
import Icon from "./Icon.jsx";

// Shown at the top of the main area when /api/health cannot be reached.
export default function OfflineBanner({ onRetry }) {
  return (
    <div role="alert" className="mb-6 flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
      <Icon name="cloud_off" size={20} className="text-red-600" />
      <p className="flex-1 text-body-md text-red-700">
        Cannot reach the backend. Check that the server is running and try again.
      </p>
      <SecondaryButton icon="refresh" onClick={onRetry}>Retry</SecondaryButton>
    </div>
  );
}
