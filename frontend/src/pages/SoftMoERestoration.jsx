import { RESTORE_ENDPOINTS } from "../api.js";
import { REQUIRED_MODELS } from "../constants.js";
import useRestoreWorkspace from "../hooks/useRestoreWorkspace.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import InputPanel from "../components/InputPanel.jsx";
import ExpertWeights from "../components/ExpertWeights.jsx";
import RestoreResultCard from "../components/RestoreResultCard.jsx";
import InfoNote from "../components/InfoNote.jsx";

// Workspace 3: a gating network blends four expert branches with soft weights.
export default function SoftMoERestoration({ health, samples, onDone }) {
  const workspace = useRestoreWorkspace(RESTORE_ENDPOINTS.softMoe, onDone);

  return (
    <div>
      <PageHeader
        title="Soft Mixture-of-Experts Restoration"
        badge="Pipeline 03: Gating network"
        badgeIcon="hub"
        subtitle="A gating network blends four branches."
      />
      <ModelNotice health={health} required={REQUIRED_MODELS["soft-moe"]} />

      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <InputPanel
            workspace={workspace}
            samples={samples}
            backendOnline={health.status !== "offline"}
            buttonLabel="Blend & Restore"
            buttonIcon="hub"
          />
          <InfoNote title="Soft gating pipeline:" iconClass="text-primary">
            The gate gives every expert branch a softmax weight; the output is the weighted sum of all four
            branch outputs.
          </InfoNote>
        </div>
        <div className="flex min-w-0 flex-col gap-6">
          <ExpertWeights result={workspace.result} loading={workspace.loading} />
          <RestoreResultCard
            workspace={workspace}
            outputLabel="Reconstructed output"
            filename="soft_moe_restoration.png"
            flowLabel="MoE blend"
            outputBadge={() => "MoE blend"}
            models={REQUIRED_MODELS["soft-moe"]}
          />
        </div>
      </div>
    </div>
  );
}
