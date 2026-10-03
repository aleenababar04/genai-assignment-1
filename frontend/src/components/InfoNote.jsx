import Icon from "./Icon.jsx";

// The small "info" card under the input card (Stitch: "Router Pipeline: ...").
export default function InfoNote({ title, iconClass = "text-primary", children }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-line bg-white p-4 shadow-xs">
      <Icon name="info" size={20} className={`mt-px ${iconClass}`} />
      <p className="text-body-sm leading-relaxed text-muted">
        <span className="font-semibold text-ink">{title}</span> {children}
      </p>
    </div>
  );
}
