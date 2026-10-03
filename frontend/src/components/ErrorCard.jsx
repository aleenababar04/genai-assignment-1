import Icon from "./Icon.jsx";

// Red error box. Used for failed requests, e.g. the backend's 503
// "Model task1_udae.onnx is not loaded." message.
export default function ErrorCard({ title = "Something went wrong", message, onDismiss }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-3.5">
      <Icon name="error" size={20} filled className="text-red-600" />
      <div className="min-w-0 flex-1">
        <p className="text-body-md font-semibold text-red-700">{title}</p>
        <p className="mt-0.5 font-mono text-mono-md break-words text-red-700">{message}</p>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="flex items-center rounded p-0.5 text-red-600 hover:bg-red-100"
          aria-label="Dismiss error"
          title="Dismiss"
        >
          <Icon name="close" size={18} />
        </button>
      )}
    </div>
  );
}
