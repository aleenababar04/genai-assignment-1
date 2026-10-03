import { useEffect, useRef, useState } from "react";
import { PrimaryButton, SecondaryButton } from "./Buttons.jsx";

// Size of the captured square photo. The backend resizes it to 128x128 anyway.
const CAPTURE_SIZE = 512;

// Webcam capture: live preview, "Capture" saves the centre square of the
// frame as a PNG File and calls onCapture(file). "Stop camera" turns it off.
// The camera is also stopped when this component disappears (tab or page change).
export default function WebcamCapture({ onCapture, disabled }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setRunning(false);
  }

  // Stop the camera on unmount.
  useEffect(() => stopCamera, []);

  async function startCamera() {
    setError(null);
    // getUserMedia only exists on secure pages (https or localhost).
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("The webcam is not available in this browser. Open the app on localhost or over https, or upload a photo instead.");
      return;
    }
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setRunning(true);
    } catch (err) {
      setError(cameraErrorMessage(err));
    } finally {
      setStarting(false);
    }
  }

  // Draw the centre square of the current video frame onto a canvas and save it as PNG.
  function capture() {
    const video = videoRef.current;
    const side = Math.min(video.videoWidth, video.videoHeight);
    if (!side) return; // the video has no frame yet
    const canvas = document.createElement("canvas");
    canvas.width = CAPTURE_SIZE;
    canvas.height = CAPTURE_SIZE;
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    canvas.getContext("2d").drawImage(video, sx, sy, side, side, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE);
    canvas.toBlob((blob) => {
      if (!blob) return;
      onCapture(new File([blob], "webcam_capture.png", { type: "image/png" }));
      stopCamera();
    }, "image/png");
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Live preview. Mirrored so it behaves like a mirror; the capture itself is not mirrored. */}
      <div className="relative aspect-square overflow-hidden rounded-xl border border-slate-200 bg-slate-900">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`size-full -scale-x-100 object-cover ${running ? "" : "hidden"}`}
        />
        {running ? (
          <>
            <span className="absolute top-2 left-2 rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white">Live</span>
            {/* Faint centred square crop guide */}
            <div className="pointer-events-none absolute inset-[15%] rounded-lg border border-dashed border-white/50" />
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-slate-300">
            Camera is off.
          </div>
        )}
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      {running ? (
        <div className="flex gap-2">
          <PrimaryButton onClick={capture} disabled={disabled} className="flex-1">
            <CameraIcon /> Capture
          </PrimaryButton>
          <SecondaryButton onClick={stopCamera}>Stop camera</SecondaryButton>
        </div>
      ) : (
        <PrimaryButton onClick={startCamera} loading={starting} loadingText="Starting camera..." disabled={disabled}>
          <CameraIcon /> Start camera
        </PrimaryButton>
      )}
    </div>
  );
}

// Friendly messages for the usual getUserMedia errors.
function cameraErrorMessage(err) {
  if (err.name === "NotAllowedError" || err.name === "SecurityError") {
    return "Camera permission was denied. Allow camera access in the browser settings, or upload a photo instead.";
  }
  if (err.name === "NotFoundError" || err.name === "OverconstrainedError") {
    return "No camera was found. Connect a webcam, or upload a photo instead.";
  }
  if (err.name === "NotReadableError") {
    return "The camera is being used by another application. Close it and try again.";
  }
  return `Could not start the camera (${err.message || err.name}).`;
}

function CameraIcon() {
  return (
    <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M4 8h3l2-3h6l2 3h3v11H4z" strokeLinejoin="round" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  );
}
