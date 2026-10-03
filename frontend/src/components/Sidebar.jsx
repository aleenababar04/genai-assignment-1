import { PAGES } from "../constants.js";
import SystemStatus from "./SystemStatus.jsx";

// Left sidebar (264px): logo, links to the five pages, System status card at the bottom.
// Links change the URL hash; App.jsx listens for that and shows the page.
export default function Sidebar({ currentPage, health }) {
  return (
    <aside className="flex flex-col border-b border-slate-200 bg-white p-4 lg:sticky lg:top-0 lg:h-screen lg:w-66 lg:shrink-0 lg:border-r lg:border-b-0">
      <a href="#/overview" className="mb-6 flex items-center gap-2 px-2 pt-2">
        <span className="flex size-8 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white">
          R
        </span>
        <span className="text-lg font-semibold text-slate-900">RestoreLab</span>
      </a>

      <nav className="flex flex-col gap-1">
        {PAGES.map((page) => {
          const active = page.id === currentPage;
          return (
            <a
              key={page.id}
              href={`#/${page.id}`}
              aria-current={active ? "page" : undefined}
              className={`rounded-lg px-3 py-2 text-sm font-medium ${
                active ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              {page.title}
            </a>
          );
        })}
      </nav>

      <div className="mt-6 lg:mt-auto">
        <SystemStatus health={health} />
      </div>
    </aside>
  );
}
