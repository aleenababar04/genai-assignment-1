import Icon from "./Icon.jsx";

// Page header from the Stitch screens: title + pipeline badge, subtitle, and two
// fact chips on the right ("Resolution: 128x128", "Engine: ONNX Runtime").
// `eyebrow` puts the badge above the title instead of next to it (Face-to-Sketch layout).
export default function PageHeader({ title, subtitle, badge, badgeIcon, eyebrow }) {
  const badgeEl = badge && (
    <span className="inline-flex items-center gap-1 rounded-full border border-primary/20 bg-indigo-50 px-2.5 py-0.5 font-mono text-mono-sm font-semibold tracking-wide text-primary uppercase">
      {badgeIcon && <Icon name={badgeIcon} size={13} />}
      {badge}
    </span>
  );

  return (
    <header className="mb-6 flex flex-col justify-between gap-4 border-b border-line pb-6 md:flex-row md:items-center">
      <div className="flex min-w-0 flex-col gap-1.5">
        {eyebrow && (
          <div className="flex flex-wrap items-center gap-2.5">
            {badgeEl}
            <span className="font-mono text-mono-sm text-slate-400">•</span>
            <span className="font-mono text-mono-sm text-muted uppercase">{eyebrow}</span>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-headline-lg text-ink">{title}</h1>
          {!eyebrow && badgeEl}
        </div>
        {subtitle && <p className="max-w-3xl text-body-md text-muted">{subtitle}</p>}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <FactChip icon="aspect_ratio" iconClass="text-muted" label="Resolution:" value="128×128" />
        <FactChip icon="bolt" iconClass="text-primary" label="Engine:" value="ONNX Runtime" />
      </div>
    </header>
  );
}

function FactChip({ icon, iconClass, label, value }) {
  return (
    <div className="flex items-center gap-1.5 rounded-lg border border-line bg-white px-3 py-1.5 font-mono text-mono-md text-ink shadow-xs">
      <Icon name={icon} size={16} className={iconClass} />
      <span className="text-muted">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
