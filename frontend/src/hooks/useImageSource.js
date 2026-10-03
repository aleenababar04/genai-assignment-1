import { useEffect, useState } from "react";
import { apiUrl } from "../api.js";
import { ACCEPTED_TYPES, UNSUPPORTED_MESSAGE } from "../constants.js";

// Keeps track of the image the user picked: an uploaded/captured file OR a sample.
//
// source = null, or {file, name, previewUrl} for a file, or {sampleId, name, previewUrl} for a sample.
// uploadError = null, or {name, message} when a file was rejected in the browser.
export default function useImageSource() {
  const [source, setSource] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  // Object URLs made with URL.createObjectURL must be released when no longer shown.
  useEffect(() => {
    const url = source?.file ? source.previewUrl : null;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [source]);

  // Accept a File from the upload zone or the webcam. Only PNG and JPG are allowed.
  function chooseFile(file) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setUploadError({ name: file.name, message: UNSUPPORTED_MESSAGE });
      setSource(null);
      return;
    }
    setUploadError(null);
    setSource({ file, name: file.name, previewUrl: URL.createObjectURL(file) });
  }

  // Use one of the backend's sample images ({id, name, url}).
  function chooseSample(sample) {
    setUploadError(null);
    setSource({ sampleId: sample.id, name: sample.name, previewUrl: apiUrl(sample.url) });
  }

  function clear() {
    setUploadError(null);
    setSource(null);
  }

  return { source, uploadError, chooseFile, chooseSample, clear };
}
