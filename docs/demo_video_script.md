# Demo video plan (5-7 minutes)

The brief requires the video to show: the application startup process, image uploading, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, and experiment-tracking records. Upload it to YouTube (Unlisted is fine) and put only the link in the report.

Record the screen with the Xbox Game Bar (`Win + Alt + R`) or OBS. Speak over it, or add short on-screen captions. Rehearse once.

## Before recording
- All seven ONNX files are in `models/` (check `docker compose up` shows 7 of 7 loaded on the Overview page).
- Have ready: one photo of a pet that is NOT in the dataset (e.g. from your phone), one already-corrupted image (save a noisy/blurred output from the app, or edit a photo), and one face photo; or plan to use the webcam.
- Open tabs: the GitHub repo, the W&B project `genai-a1`, the Kaggle notebook pages (optional).
- Close other windows and notifications.

## Shot list

| Time | What to show | What to say (short) |
|---|---|---|
| 0:00-0:30 | GitHub repo page, then a terminal in the project folder | "This is the repository. The models are downloaded into `models/`, and the whole app starts with one command." |
| 0:30-1:15 | Run `docker compose down` then `docker compose up --build`; wait for "healthy"; open http://localhost:8080 | "The backend loads the seven ONNX models; the frontend is served by nginx." Point at the System status card: 7 of 7 loaded. |
| 1:15-1:30 | Overview page | "Four workspaces, one for each task." |
| 1:30-2:30 | **Universal Restoration**: upload your own pet photo, choose Salt-and-pepper / High, Restore; then pick a sample, choose Occlusion / Medium, Restore; click Download result | "Runtime corruption is applied in the backend, the autoencoder restores it. Settings, inference time, download." |
| 2:30-3:30 | **Hard-Routed Restoration**: same photo with Gaussian blur / High; show the four probabilities, predicted corruption and selected expert; then corruption None to show "Identity bypass"; then upload the already-corrupted image with corruption None | "The classifier picks one specialist; a clean image skips the experts entirely." |
| 3:30-4:30 | **Soft Mixture-of-Experts Restoration**: occlusion, then salt-and-pepper; point at the stacked weight bar and the top contributor | "Every branch gets a weight; the output is the weighted sum." |
| 4:30-5:30 | **Face-to-Sketch Generator**: upload a face (or webcam capture), generate Style 1, Style 2, Style 3; download the sketch | "The style is a learned embedding fed to the generator." |
| 5:30-6:30 | W&B project: the Optuna trial runs of one task, a final run's charts and logged sample images, the artifacts tab; optionally an Optuna plot from the report | "All trials, losses, evaluation results, checkpoints and sample images are tracked in Weights & Biases." |
| 6:30-6:50 | Back to the app or repo | One-sentence wrap-up. |
