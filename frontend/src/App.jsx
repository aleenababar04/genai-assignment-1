import { useCallback, useEffect, useState } from "react";
import { getHealth, getSamples } from "./api.js";
import { PAGES } from "./constants.js";
import Sidebar from "./components/Sidebar.jsx";
import OfflineBanner from "./components/OfflineBanner.jsx";
import Overview from "./pages/Overview.jsx";
import UniversalRestoration from "./pages/UniversalRestoration.jsx";
import HardRoutedRestoration from "./pages/HardRoutedRestoration.jsx";
import SoftMoERestoration from "./pages/SoftMoERestoration.jsx";
import FaceToSketch from "./pages/FaceToSketch.jsx";

const HEALTH_POLL_MS = 10000;

// Navigation: the current page is stored in the URL hash, e.g. "#/universal".
// This keeps the browser back button and page reloads working without a router library.
function readPageFromHash() {
  const id = window.location.hash.replace(/^#\/?/, "");
  return PAGES.some((page) => page.id === id) ? id : "overview";
}

function useHashPage() {
  const [page, setPage] = useState(readPageFromHash);
  useEffect(() => {
    const onChange = () => setPage(readPageFromHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return page;
}

// Backend health, polled every 10 seconds.
// status: "checking" (first request pending), "online" or "offline".
function useHealth() {
  const [health, setHealth] = useState({ status: "checking", data: null });

  const refresh = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth({ status: "online", data });
    } catch {
      setHealth({ status: "offline", data: null });
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, HEALTH_POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  return [health, refresh];
}

// Sample images, loaded once (and again when the backend comes back online).
function useSamples(online) {
  const [samples, setSamples] = useState([]);
  useEffect(() => {
    if (!online) return;
    getSamples()
      .then(setSamples)
      .catch(() => setSamples([]));
  }, [online]);
  return samples;
}

export default function App() {
  const page = useHashPage();
  const [health, refreshHealth] = useHealth();
  const samples = useSamples(health.status === "online");

  // Props every workspace receives. onDone refreshes the status card after a run,
  // so "Last inference" updates straight away.
  const workspaceProps = { health, samples, onDone: refreshHealth };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 lg:flex">
      <Sidebar currentPage={page} health={health} />

      <main className="min-w-0 flex-1 p-6 lg:p-8">
        {health.status === "offline" && <OfflineBanner onRetry={refreshHealth} />}

        {page === "overview" && <Overview health={health} />}
        {page === "universal" && <UniversalRestoration {...workspaceProps} />}
        {page === "hard-routed" && <HardRoutedRestoration {...workspaceProps} />}
        {page === "soft-moe" && <SoftMoERestoration {...workspaceProps} />}
        {page === "face-to-sketch" && <FaceToSketch {...workspaceProps} />}
      </main>
    </div>
  );
}
