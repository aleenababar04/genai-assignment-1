// The white card used everywhere: 1px border, 12px radius, soft shadow, 24px padding.
// Change the look of every card in the app here.
export default function Card({ title, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-6 shadow-sm ${className}`}>
      {title && <h2 className="mb-4 text-base font-semibold text-slate-900">{title}</h2>}
      {children}
    </section>
  );
}
