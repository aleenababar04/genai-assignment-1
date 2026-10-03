import { useState } from "react";
import { restoreImage } from "../api.js";
import useImageSource from "./useImageSource.js";

// All the state of one restoration workspace. The three restoration pages use
// this hook and differ only in the endpoint and in how they show the result.
export default function useRestoreWorkspace(endpoint, onDone) {
  const image = useImageSource();
  const [corruption, setCorruption] = useState("salt_pepper");
  const [severity, setSeverity] = useState("medium");
  const [seed, setSeed] = useState(""); // optional; empty = backend picks a random seed
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // backend JSON of the last successful run
  const [error, setError] = useState(null); // message of the last failed run

  async function run() {
    if (!image.source) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await restoreImage(endpoint, image.source, { corruption, severity, seed });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      onDone?.(); // refresh the System status card
    }
  }

  return {
    image,
    corruption, setCorruption,
    severity, setSeverity,
    seed, setSeed,
    loading, result, error,
    clearError: () => setError(null),
    run,
  };
}
