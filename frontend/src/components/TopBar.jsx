import Icon from "./Icon.jsx";
import { StatusDot } from "./SystemStatus.jsx";

// Top navigation bar of the Stitch screens: logo + product name, the "Studio" tab,
// and (on the right) the live backend status.
export default function TopBar({ health }) {
  return (
    <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center justify-between border-b border-line bg-white px-6 shadow-sm">
      <div className="flex h-full items-center gap-8">
        <a href="#/overview" className="flex items-center gap-2.5">
          <span className="flex size-7 items-center justify-center rounded-lg bg-primary text-white shadow-sm">
            <Icon name="csv" size={18} />
          </span>
          <span className="text-headline-sm font-bold tracking-tight text-ink">RestoreLab</span>
        </a>
        <nav className="hidden h-full items-center md:flex">
          <span className="flex h-full items-center border-b-2 border-primary text-body-md font-semibold text-primary">
            Studio
          </span>
        </nav>
      </div>
      <div className="flex items-center gap-2 rounded-lg border border-line bg-page px-3 py-1.5">
        <StatusDot status={health.status} compact />
        <Icon name="dns" size={16} className="text-muted" />
      </div>
    </header>
  );
}

// Footer line at the bottom of every page.
export function Footer() {
  return (
    <footer className="mt-8 flex flex-col items-center justify-between gap-3 border-t border-line pt-4 font-mono text-mono-sm text-muted sm:flex-row">
      <div className="flex items-center gap-2">
        <Icon name="science" size={16} />
        <span>RestoreLab Computational Imaging Laboratory</span>
      </div>
      <div className="flex items-center gap-4">
        <span>ONNX Runtime</span>
        <span>•</span>
        <span>128×128 pipeline</span>
        <span>•</span>
        <span className="font-medium text-primary">FastAPI backend</span>
      </div>
    </footer>
  );
}
