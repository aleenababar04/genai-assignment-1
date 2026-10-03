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
      <PageHeader title="Universal Restoration" subtitle="One autoencoder restores any corruption." />
      <ModelNotice health={health} required={REQUIRED_MODELS.universal} />

      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        <InputPanel workspace={workspace} samples={samples} backendOnline={health.status !== "offline"} />
        <div className="min-w-0 flex-1">
          <RestoreResultCard
            workspace={workspace}
            outputLabel="Restored output"
            filename="universal_restoration.png"
          />
        </div>
      </div>
    </div>
  );
}
