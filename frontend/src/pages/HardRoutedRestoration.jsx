import { RESTORE_ENDPOINTS } from "../api.js";
import { REQUIRED_MODELS } from "../constants.js";
import useRestoreWorkspace from "../hooks/useRestoreWorkspace.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import InputPanel from "../components/InputPanel.jsx";
import ProbabilityBars from "../components/ProbabilityBars.jsx";
import RestoreResultCard from "../components/RestoreResultCard.jsx";

// Workspace 2: a classifier predicts the corruption and ONE specialist restores it.
export default function HardRoutedRestoration({ health, samples, onDone }) {
  const workspace = useRestoreWorkspace(RESTORE_ENDPOINTS.hardRouting, onDone);

  return (
    <div>
      <PageHeader
        title="Hard-Routed Restoration"
        subtitle="A classifier predicts the corruption and routes the image to one specialist."
      />
      <ModelNotice health={health} required={REQUIRED_MODELS["hard-routed"]} />

      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        <InputPanel workspace={workspace} samples={samples} backendOnline={health.status !== "offline"} />
        <div className="flex min-w-0 flex-1 flex-col gap-6">
          <ProbabilityBars result={workspace.result} loading={workspace.loading} />
          <RestoreResultCard
            workspace={workspace}
            outputLabel="Reconstructed output"
            filename="hard_routed_restoration.png"
          />
        </div>
      </div>
    </div>
  );
}
