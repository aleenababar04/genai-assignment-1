import { RESTORE_ENDPOINTS } from "../api.js";
import { REQUIRED_MODELS } from "../constants.js";
import useRestoreWorkspace from "../hooks/useRestoreWorkspace.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import InputPanel from "../components/InputPanel.jsx";
import ExpertWeights from "../components/ExpertWeights.jsx";
import RestoreResultCard from "../components/RestoreResultCard.jsx";

// Workspace 3: a gating network blends four expert branches with soft weights.
export default function SoftMoERestoration({ health, samples, onDone }) {
  const workspace = useRestoreWorkspace(RESTORE_ENDPOINTS.softMoe, onDone);

  return (
    <div>
      <PageHeader
        title="Soft Mixture-of-Experts Restoration"
        subtitle="A gating network blends four branches."
      />
      <ModelNotice health={health} required={REQUIRED_MODELS["soft-moe"]} />

      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        <InputPanel workspace={workspace} samples={samples} backendOnline={health.status !== "offline"} />
        <div className="flex min-w-0 flex-1 flex-col gap-6">
          <ExpertWeights result={workspace.result} loading={workspace.loading} />
          <RestoreResultCard
            workspace={workspace}
            outputLabel="Reconstructed output"
            filename="soft_moe_restoration.png"
          />
        </div>
      </div>
    </div>
  );
}
