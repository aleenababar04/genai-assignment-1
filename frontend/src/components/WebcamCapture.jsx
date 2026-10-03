import { useEffect, useRef, useState } from "react";
import { PrimaryButton, SecondaryButton } from "./Buttons.jsx";
import Icon from "./Icon.jsx";

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
      <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-line bg-slate-900 shadow-inner">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`size-full -scale-x-100 object-cover ${running ? "" : "hidden"}`}
        />
        {running ? (
          <>
            <span className="absolute top-2.5 left-2.5 flex items-center gap-1.5 rounded-full bg-red-600 px-2 py-0.5 text-[11px] font-semibold text-white shadow-sm">
              <span className="size-2 animate-pulse rounded-full bg-white" />
              Live
            </span>
            <span className="absolute top-2.5 right-2.5 rounded border border-white/20 bg-black/50 px-2 py-0.5 font-mono text-mono-sm text-white backdrop-blur-xs">
              Mirrored preview
            </span>
            {/* Crop guide: the captured centre square, scaled to 128 x 128 by the backend */}
            <div className="pointer-events-none absolute inset-[8%] flex flex-col items-center justify-between rounded-lg border-2 border-dashed border-white/80 p-1.5">
              <span className="rounded bg-black/60 px-1 font-mono text-[9px] text-white/90">Captured area → 128 × 128</span>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
            <span className="flex size-12 items-center justify-center rounded-full bg-white/10 text-slate-300">
              <Icon name="videocam_off" size={24} />
            </span>
            <p className="text-body-sm font-medium text-slate-200">Camera is off.</p>
            <p className="font-mono text-mono-sm text-slate-400">Start the camera to take a photo</p>
          </div>
        )}
      </div>

      {error && (
        <p className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-body-sm text-red-700">
          <Icon name="error" size={16} className="mt-px text-red-600" />
          {error}
        </p>
      )}

      {running ? (
        <div className="flex gap-2">
          <PrimaryButton icon="photo_camera" onClick={capture} disabled={disabled} size="sm" className="flex-1">
            Capture
          </PrimaryButton>
          <SecondaryButton icon="videocam_off" onClick={stopCamera} className="h-9">
            Stop camera
          </SecondaryButton>
        </div>
      ) : (
        <PrimaryButton icon="photo_camera" onClick={startCamera} loading={starting} loadingText="Starting camera..." disabled={disabled} size="sm">
          Start camera
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
