import { RESTORE_ENDPOINTS } from "../api.js";
import { REQUIRED_MODELS } from "../constants.js";
import useRestoreWorkspace from "../hooks/useRestoreWorkspace.js";
import PageHeader from "../components/PageHeader.jsx";
import ModelNotice from "../components/ModelNotice.jsx";
import InputPanel from "../components/InputPanel.jsx";
import RestoreResultCard from "../components/RestoreResultCard.jsx";

// Workspace 1: one denoising autoencoder for every corruption.
export default function UniversalRestoration({ health, samples, onDone }) {
  const workspace = useRestoreWorkspace(RESTORE_ENDPOINTS.universal, onDone);

  return (
    <div>
      <PageHeader
        title="Universal Restoration"
        badge="Pipeline 01: All corruptions"
        badgeIcon="layers"
        subtitle="One autoencoder restores any corruption: a single network trained on salt-and-pepper noise, Gaussian blur and rectangular occlusion."
      />
      <ModelNotice health={health} required={REQUIRED_MODELS.universal} />

      <div className="grid items-start gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <InputPanel
          workspace={workspace}
          samples={samples}
          backendOnline={health.status !== "offline"}
          buttonLabel="Restore Tensor"
          buttonIcon="auto_fix_high"
        />
        <RestoreResultCard
          workspace={workspace}
          outputLabel="Restored output"
          filename="universal_restoration.png"
          flowLabel="Restore"
          outputBadge={() => "Autoencoder"}
          models={REQUIRED_MODELS.universal}
        />
      </div>
    </div>
  );
}
