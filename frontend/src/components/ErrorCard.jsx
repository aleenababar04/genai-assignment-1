// Red error box. Used for failed requests, e.g. the backend's 503
// "Model task1_udae.onnx is not loaded." message.
export default function ErrorCard({ title = "Something went wrong", message, onDismiss }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4">
      <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-red-600 text-xs font-bold text-white">
        !
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-red-700">{title}</p>
        <p className="mt-1 text-sm break-words text-red-700">{message}</p>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="text-sm text-red-700 hover:underline"
          aria-label="Dismiss error"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
