import Icon from "./Icon.jsx";

// The white card used everywhere (Stitch: 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding).
// With a `title` it gets the Stitch card header: title + optional subtitle on the left,
// an optional icon or `aside` (badges) on the right, and a divider below.
export default function Card({ title, subtitle, icon, aside, children, className = "" }) {
  return (
    <section className={`flex flex-col gap-5 rounded-xl border border-line bg-white p-6 shadow-xs ${className}`}>
      {title && (
        <div className={`flex items-center justify-between gap-3 border-b border-line pb-3 ${icon ? "" : "flex-wrap"}`}>
          <div className="min-w-0">
            <h2 className="text-headline-sm text-ink">{title}</h2>
            {subtitle && <p className="mt-0.5 text-body-sm text-muted">{subtitle}</p>}
          </div>
          {aside}
          {icon && <Icon name={icon} size={20} className="text-muted" />}
        </div>
      )}
      {children}
    </section>
  );
}

// Small uppercase section label inside a card ("OR PICK A SAMPLE", "CORRUPTION TYPE").
export function SectionLabel({ children, className = "" }) {
  return <span className={`text-label-ui font-semibold tracking-wider text-muted uppercase ${className}`}>{children}</span>;
}
