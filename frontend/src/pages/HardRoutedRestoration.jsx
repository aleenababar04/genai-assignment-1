import { RESTORE_ENDPOINTS } from "../api.js";
import { CATEGORIES, REQUIRED_MODELS } from "../constants.js";
import useRestoreWorkspace from "../hooks/useRestoreWorkspace.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import InputPanel from "../components/InputPanel.jsx";
import ProbabilityBars from "../components/ProbabilityBars.jsx";
import RestoreResultCard from "../components/RestoreResultCard.jsx";
import InfoNote from "../components/InfoNote.jsx";

// Workspace 2: a classifier predicts the corruption and ONE specialist restores it.
export default function HardRoutedRestoration({ health, samples, onDone }) {
  const workspace = useRestoreWorkspace(RESTORE_ENDPOINTS.hardRouting, onDone);

  return (
    <div>
      <PageHeader
        title="Hard-Routed Restoration"
        badge="Pipeline 02: Classifier directed"
        badgeIcon="alt_route"
        subtitle="A classifier predicts the corruption and routes the image to one specialist."
      />
      <ModelNotice health={health} required={REQUIRED_MODELS["hard-routed"]} />

      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <InputPanel
            workspace={workspace}
            samples={samples}
            backendOnline={health.status !== "offline"}
            buttonLabel="Route & Restore"
            buttonIcon="alt_route"
          />
          <InfoNote title="Router pipeline:" iconClass="text-cyan-500">
            The classifier scores the four classes and the image goes to the single specialist with the highest
            probability. Images predicted clean are returned unchanged (identity bypass).
          </InfoNote>
        </div>
        <div className="flex min-w-0 flex-col gap-6">
          <ProbabilityBars result={workspace.result} loading={workspace.loading} />
          <RestoreResultCard
            workspace={workspace}
            outputLabel="Reconstructed output"
            filename="hard_routed_restoration.png"
            flowLabel="Route"
            outputBadge={(result) =>
              result.identity_bypass ? "Identity bypass" : CATEGORIES[result.selected_expert]?.expert ?? result.selected_expert
            }
            models={REQUIRED_MODELS["hard-routed"]}
          />
        </div>
      </div>
    </div>
  );
}
