# Testing guide

How to check every part of the project yourself. Commands run in PowerShell from the project folder.

## 1. Automated tests (2 minutes)

```powershell
.venv\Scripts\python.exe -m pytest -q                 # training code: expect "183 passed"
.venv\Scripts\python.exe -m pytest backend/tests -q   # backend: expect "73 passed"
```

These check the building blocks: corruptions, data split, models, losses, routing, ONNX export, API validation.

## 2. Start the app

```powershell
docker compose up --build -d        # -d = in the background
```

Open http://localhost:8080. The sidebar's status card should say **Backend online** and how many of the 7 models are loaded. After copying new model files into `models/`, run `docker compose restart backend`. To stop everything: `docker compose down`.

## 3. Test each workspace by hand

Use the six sample pets and, more importantly, **your own photos** (the evaluator will bring unseen images). Any PNG/JPG works; it is resized to 128x128.

### Universal Restoration (Task 1)
| Do this | You should see |
|---|---|
| Pick a sample, corruption **Salt-and-pepper**, **High**, Restore | Noisy input on the left, clean but slightly smooth output on the right; settings chips (probability 0.15); inference time; Download works |
| Same with **Occlusion / Medium** | Two black boxes in the input; the output fills them with blurry colours from around them |
| Same with **Gaussian blur / Low** and **None** | Output slightly *smoother* than the input. Expected: the bottleneck limits detail (see report) |
| Upload your own pet photo, any corruption | Works the same way |
| Upload a photo with a large black background, corruption None | Model may "fill in" the black area. Known failure case, in the report |

### Hard-Routed Restoration (Task 2)
| Do this | You should see |
|---|---|
| Sample, **Blur / High** | Four probability bars, Blur near 100%, "Selected expert: Blur" |
| Sample, **None** | Clean near 100%, "Identity bypass", output identical to input |
| Each corruption at each severity | The predicted corruption matches what you chose (classifier is about 99% accurate) |
| **Occlusion / Low** a few times with different seeds | Occasionally predicted "clean" (the classifier's most common mistake) |
| Upload an image that is already corrupted (e.g. download a noisy input from Universal, then upload it with corruption None) | The classifier detects the noise and routes it to the salt-and-pepper expert |

### Soft Mixture-of-Experts Restoration (Task 3)
Until the Task 3 model is trained this page shows "model not loaded"; that is the expected behaviour. Afterwards:
| Do this | You should see |
|---|---|
| Each corruption | Four weights summing to 1, the matching expert highlighted as top contributor |
| None | Most weight on Identity |

### Face-to-Sketch Generator (Task 4)
| Do this | You should see |
|---|---|
| Upload a face photo, Style 1, Generate | Photo and sketch side by side, inference time, Download sketch |
| Same photo with Style 2 and Style 3 | Visibly different sketches (the style condition works) |
| Use webcam, Capture, Generate | Same, from the camera (allow camera access in the browser) |
| A non-face photo | Still produces a sketch-like image, usually poor: the model only knows faces |

### Error handling
| Do this | You should see |
|---|---|
| Upload a PDF or text file | Red "Unsupported file type" message, no crash |
| Stop the backend (`docker compose stop backend`) and click a button | "Backend offline" banner; restart with `docker compose start backend` |

## 4. Look at the results
All tables are in `report/results/` and all figures in `report/figures/` (open the PNGs in VS Code or Explorer):
- Task 1: `task1_udae_examples.png`, `task1_udae_failures.png`, `task1_udae_curves.png`, `task1_udae_optuna.png`
- Task 2: `task2_test_confusion.png`, `task2_examples.png`, `task2_misrouted.png`
- Task 3 / Task 4: after their evaluation runs.

## 5. Experiment tracking
On wandb.ai, project `genai-a1`: one group per model, `trial-...` runs (Optuna), `final-...` runs (charts, sample images, the checkpoint under Artifacts), `eval-...` runs (test tables and figures).

## 6. Final check before submitting: the evaluator's view
Once the models are published as the GitHub Release, test exactly what the evaluator will do, in a NEW folder:

```powershell
cd $HOME\Desktop
git clone https://github.com/aleenababar04/genai-assignment-1.git eval-test
cd eval-test
python scripts/download_models.py
docker compose down   # (run this in the ORIGINAL project folder first, so port 8080 is free)
docker compose up --build
```

Open http://localhost:8080 and check that all 7 models load and every workspace works.
