# Testing and outputs guide

Two things in one place: **how to test the whole project yourself**, and **where every output is and what it means**. Commands are for PowerShell, run from the project folder.

Fastest way to see everything:

```powershell
.venv\Scripts\python.exe scripts\show_results.py        # every key number, in plain text
.venv\Scripts\python.exe scripts\build_gallery.py       # then open report\outputs_gallery.html in a browser
```

The gallery shows every figure with a plain-English "what it shows / how to read it / what it proves".

## Contents

1. [Automated tests](#1-automated-tests)
2. [Start the application](#2-start-the-application)
3. [Manual test plan, workspace by workspace](#3-manual-test-plan)
4. [Tests with unseen images and error cases](#4-unseen-images-and-error-cases)
5. [Where every output is, and what it means](#5-where-every-output-is-and-what-it-means)
6. [Test it the way the evaluator will](#6-test-it-the-way-the-evaluator-will)
7. [Sign-off checklist](#7-sign-off-checklist)

---

## 1. Automated tests

```powershell
.venv\Scripts\python.exe -m pytest -q                  # training code: expect "184 passed"
.venv\Scripts\python.exe -m pytest backend/tests -q    # backend:       expect "73 passed"
```

A few warnings about the "legacy ONNX exporter" are normal. **Pass = no `failed` or `error` in the last line.**

| Tests in | They check | Example of what would break them |
|---|---|---|
| `tests/test_corruptions.py` | Each corruption behaves as specified: salt-and-pepper changes about the right fraction of pixels, occlusion covers 10 to 35% without overlap, the same seed gives the same result | Changing a range constant in `corruptions.py` |
| `tests/test_manifests.py`, `test_prepare_pet.py` | Splits are 80/20 and reproducible; test manifest has 10 entries per image with the fixed severities | Changing the split seed |
| `tests/test_autoencoder.py` | Output shape and range, bottleneck is smaller than the input, ONNX export matches PyTorch | Editing a layer size |
| `tests/test_losses_metrics.py` | L1, SSIM, PSNR and the summary tables compute correctly | Editing a metric |
| `tests/test_classifier_balanced.py` | Every batch is exactly 25% per class | Editing `balanced.py` |
| `tests/test_hard_router.py` | Clean images come back bit-identical; an expert is never run when nothing is routed to it | Editing the routing |
| `tests/test_soft_moe.py` | The mixture equals the weighted sum; frozen experts get no gradients and keep BatchNorm statistics; one-graph ONNX export matches | Editing `soft_moe.py` |
| `tests/test_cgan.py`, `test_fs2k.py` | Style changes the output of both networks; photo and sketch get identical augmentation | Editing the GAN or the augmentation |
| `backend/tests/` | Upload validation (type, size, real image), every endpoint, error codes, OpenCV blur equals PyTorch blur | Editing the API |

## 2. Start the application

```powershell
docker compose up --build -d          # -d keeps it running in the background
docker compose ps                     # both services should say "Up"; the backend "healthy"
```

Open **http://localhost:8080**. The status card in the sidebar must show **Backend online** and **7 of 7 loaded**.

Handy while testing:

| Command | What it shows |
|---|---|
| `docker compose logs -f backend` | A live log of every request the backend handles (`Ctrl+C` to stop watching) |
| http://localhost:8080/api/health | The backend's own report: which of the 7 model files loaded, last inference time |
| `docker compose restart backend` | Reloads the models (after copying new files into `models/`) |
| `docker compose down` | Stops and removes the containers |

## 3. Manual test plan

Use the six sample pets, and **your own photos** too. Tick each row.

### 3.1 Universal Restoration (Task 1)

| # | Do this | Expected | Pass |
|---|---|---|---|
| 1 | Pick a sample, corruption **Salt-and-pepper**, severity **High**, press Restore | Left: image full of black/white dots. Right: clean but slightly smooth. Chips show "Noise probability: 0.15". Inference time about 40 to 100 ms | [ ] |
| 2 | **Occlusion / Medium** | Left: two black boxes. Right: boxes filled with blurry colours from the surroundings (it guesses; it cannot know what was hidden) | [ ] |
| 3 | **Gaussian blur / High** | Output slightly sharper but still soft. Little improvement is expected (blur already removed the detail) | [ ] |
| 4 | **None** | Output is slightly *smoother* than the input. This is the known weakness (the bottleneck loses fine detail) | [ ] |
| 5 | Press **Download result** | A PNG file downloads and opens | [ ] |
| 6 | Type a seed (for example 7), restore twice | Identical result both times. With the field empty, the corruption differs each time | [ ] |

### 3.2 Hard-Routed Restoration (Task 2)

| # | Do this | Expected | Pass |
|---|---|---|---|
| 1 | Sample, **Blur / High** | Four bars: Blur near 100%. "Predicted: blur", "Selected expert: Blur" | [ ] |
| 2 | **Salt-and-pepper** (any severity) | Salt-and-pepper 100% | [ ] |
| 3 | **None** | Clean near 100%, "Identity bypass", output identical to the input | [ ] |
| 4 | **Occlusion / Low**, seed 1, 2, 3, 4, 5 | Usually "occlusion". Occasionally "clean": the classifier's most common mistake (2.8% of the time) | [ ] |
| 5 | Download a noisy result from the Universal page, then upload it here with corruption **None** | The classifier detects the noise by itself and routes to the salt-and-pepper expert | [ ] |

### 3.3 Soft Mixture-of-Experts Restoration (Task 3)

| # | Do this | Expected | Pass |
|---|---|---|---|
| 1 | **Salt-and-pepper / High** | Weights about 1.00 on salt-and-pepper | [ ] |
| 2 | **Blur / Low**, then **High** | Low: weight shared between identity (about 0.4) and blur (about 0.6). High: blur about 0.95. The routing sharpens with severity | [ ] |
| 3 | **Occlusion / Low**, then **High** | Low: identity often wins (about 0.6). High: occlusion about 0.99 | [ ] |
| 4 | **None** | Identity about 0.98. The stacked bar is almost entirely the identity colour | [ ] |
| 5 | Check the four weights add up to 1 | Yes (shown rounded) | [ ] |

### 3.4 Face-to-Sketch Generator (Task 4)

| # | Do this | Expected | Pass |
|---|---|---|---|
| 1 | Upload a face photo, **Style 1**, Generate | Photo and a sketch side by side. About 40 ms | [ ] |
| 2 | Same photo, **Style 2**, then **Style 3** | Three clearly different drawings: Style 1 light outline, Style 2 heavy dark shading, Style 3 soft toned | [ ] |
| 3 | **Use webcam**, allow the camera, Capture, Generate | Sketch of your own face | [ ] |
| 4 | Download the sketch | A PNG downloads | [ ] |
| 5 | Upload a photo that is not a face | A sketch-like image, usually poor: the model only knows faces | [ ] |

### 3.5 Overview and status

| # | Check | Pass |
|---|---|---|
| 1 | Overview shows four workspace cards and a model list with "Loaded" for all 7 | [ ] |
| 2 | Each card's "Open workspace" button works; the sidebar highlights the current page | [ ] |
| 3 | After running a model, "Last inference" in the status card updates | [ ] |

## 4. Unseen images and error cases

The evaluator will bring their own images.

| Test | Expected |
|---|---|
| A photo from your phone (large, JPG) of a pet | Works; resized to 128 x 128 (non-square photos are squashed, which is expected) |
| A photo of a **person or a car** on the restoration pages | Works technically; quality is poor because the models only saw cats and dogs |
| A cat on a **black background** (clean) on Universal and Hard-Routed | Reproduces the failure case in the report: black area "repaired" or the image routed to the occlusion expert |
| A **PDF or .txt** file | Red "Unsupported file type. Upload a PNG or JPG image." No crash |
| A **file over 10 MB** | Rejected with a "File too large" message |
| A renamed file (for example a `.txt` renamed to `.png`) | Rejected ("Could not read the image") |
| Stop the backend: `docker compose stop backend`, then press Restore | The "Backend offline" banner or an error card appears; restart with `docker compose start backend` |
| Delete one model file from `models/`, `docker compose restart backend` | That workspace shows "Model ... is not loaded"; the others still work (put the file back afterwards) |

## 5. Where every output is, and what it means

### 5.1 The big picture

| Folder | Contains | Open with |
|---|---|---|
| `report/results/` | Result tables (`.csv`) and ONNX checks (`.json`) | Excel, or VS Code, or Notepad |
| `report/figures/` | All pictures: examples, curves, confusion matrix, heatmaps | Any image viewer, or the gallery |
| `report/figures/stitch/` | The Google Stitch designs (screenshots and HTML) | Image viewer / browser |
| `optuna_studies/` | One database per hyperparameter search | `scripts/show_results.py` |
| `manifests/` | The fixed data splits and fixed corruptions | Notepad (text) |
| `configs/` | Settings; `*_best.yaml` = what Optuna chose | Notepad |
| `models/` | The 7 ONNX files the app uses | (binary) |
| `checkpoints/` | PyTorch weights from training (bigger; used for evaluation) | (binary) |
| W&B (website) | Every run, curve, sample image, checkpoint | Browser |

### 5.2 The result tables (`report/results/`)

Every restoration table has the same columns: `condition`, `severity`, `count` (images), `psnr` and `ssim` (output vs. the clean image), `input_psnr` and `input_ssim` (the damaged input vs. the clean image: **the "do nothing" baseline**), and `l1`. The rows `corrupted / all` and `all / all` are averages. **Always compare `psnr` with `input_psnr`: improvement is the point.** A clean input has infinite input PSNR (shown as 100 in the file).

| File | What it is | Key thing to read |
|---|---|---|
| `task1_udae_test_metrics.csv` | Task 1 on the test set, per condition and severity | The `corrupted` row: 19.25 -> 23.96 dB |
| `task1_udae_skip_test_metrics.csv` | The skip-connection experiment | Same row: 26.77 dB (better, but we keep the no-skip model) |
| `task2_classifier_per_class.csv` | Precision, recall, F1 for each of the four classes | All above 0.97; salt-and-pepper is 1.000 |
| `task2_classifier_by_severity.csv` | Classifier accuracy per condition and severity, with the most common wrong answer | Only low occlusion (0.972) and clean (0.990) are below 0.999 |
| `task2_oracle_test_metrics.csv`, `task2_predicted_test_metrics.csv` | Restoration with the true label choosing the expert, and with the classifier choosing it | They are almost identical: the classifier is accurate enough |
| `task2_misrouting.csv` | The 143 misrouted images by (true class, branch chosen) with the quality change | 101 low-occlusion images sent to "identity" cost nothing |
| `task3_moe_test_metrics.csv` | Task 3 restoration | Clean 62 dB (almost untouched); corrupted 23.71 dB |
| `task3_moe_mean_weights.csv` | **The gate's average weights** per true corruption and severity | Each row's biggest number is the expert used |
| `task3_moe_expert_health.csv` | Is any expert unused or dominating? | All `inactive` and `dominates_unrelated` are False |
| `task4_cgan_test_metrics.csv` | Sketch quality overall and per style (200-epoch model; `task4_cgan_100ep_test_metrics.csv` is the earlier 100-epoch model) | SSIM 0.473; Style 2 lowest |
| `*_onnx_check.json` | PyTorch output vs ONNX output on real images | `max_abs_diff` about 1e-6; the limit is 1e-4 |

### 5.3 The figures (`report/figures/`)

The gallery explains all of them. The essentials:

| Figure | How to read it | What it shows |
|---|---|---|
| `corruption_preview.png` | Each row is one test image: clean, then each corruption at low, medium, high | The exact damage used in testing |
| `task1_udae_examples.png` | 12 rows: clean target, corrupted input, restored output, **error map** (bright = large error) | Restoration quality; where mistakes concentrate (edges, filled boxes) |
| `task1_udae_failures.png` | The worst test image of each condition | The black-background cat and the dense flower field: the model's limits |
| `task1_udae_psnr_bars.png`, `..._ssim_bars.png` | Blue = damaged input, orange = restored output, per condition and severity | Orange above blue = it helped; below blue (clean, mild blur) = it hurt |
| `task1_udae_curves.png` | Training loss and validation score per epoch | Steady improvement, no overfitting |
| `task1_udae_optuna.png` (also task2 and task3 and task4) | Left: every trial's score (x = pruned) and the best so far. Right: which settings mattered | The search worked; the learning rate mattered most |
| `task1_udae_skip_*` | The same figures for the skip model | The skip makes outputs sharper |
| `task2_test_confusion.png` | Rows = true class, columns = predicted; a perfect classifier is a bright diagonal | The classifier is almost perfect |
| `task2_misrouted.png` | The six worst misrouted images | All are clean photos with big black areas |
| `task2_examples.png` | 12 examples with the chosen expert in each row label | Hard routing in action |
| `task3_moe_routing_heatmap.png` | Rows = true corruption and severity, columns = the four branches, brightness = average weight | **The key Task 3 figure**: a bright diagonal, getting brighter with severity |
| `task3_moe_weight_distribution.png` | Box plots of each branch's weight for each true corruption | The spread across images, not just the average |
| `task3_moe_dominant_examples.png`, `..._distributed_examples.png` | Images where one expert wins, and where the weight is shared | When the gate is sure and when it hedges |
| `task3_moe_failures.png` | Worst cases with the four weights in the labels | Tells routing errors from expert errors |
| `task4_cgan_style_swap.png` | The same face in Styles 1, 2, 3 | **Proof that the style condition works** |
| `task4_cgan_examples.png`, `task4_cgan_failures.png` | Photo, true sketch, generated sketch | Typical results, and the four worst (all Style 2, long hair) |
| `task4_cgan_curves.png` | Discriminator losses, generator losses, validation score | A healthy GAN: nothing collapsed; validation flat after about 100 epochs |

### 5.4 The Optuna studies (`optuna_studies/`)

Five `.db` files (SQLite): `task1_udae`, `task2_classifier`, `task2_specialists`, `task3_moe`, `task4_cgan`. Each stores every trial: its settings, its scores per epoch, and whether it completed or was pruned (stopped early because it was worse than the median).

- Summary of all five: `python scripts\show_results.py` (the OPTUNA section: trials, how many completed and pruned, the best trial and its settings).
- Pictures of each: `report/figures/task*_optuna.png`.
- The chosen settings are copied into `configs/*_best.yaml`; those are what the final training used.
- Optional: install `optuna-dashboard` and run `optuna-dashboard sqlite:///optuna_studies/task1_udae.db` for an interactive view.

### 5.5 Weights & Biases (the website)

Open https://wandb.ai, your account, project **genai-a1**. Each task has a *group*; click a group to see its runs:

| Run name | What it is | What to click |
|---|---|---|
| `trial-000`, `trial-001`, ... | One Optuna trial each | Open one to see its loss curves. In the project table, sort by the validation score column |
| `final-task1_udae` (and `final-task1_udae_skip`, `final-task2_classifier`, `final-task2_specialist_*`, `final-task3_moe`, `final-task4_cgan`) | The full training of the chosen configuration | **Charts** tab: train and validation curves. **Media** tab or the "samples" panel: images saved during training (for Task 4, the same faces every 10 epochs). **Artifacts** tab: the saved checkpoint |
| `eval-...` | The test-set evaluation | Tables with the results and the example figures |

Use this tab during the video to satisfy "experiment-tracking records".

### 5.6 The other files

- `manifests/pet_split.json`: lists of image names for train, validation and test. `pet_val_manifest.jsonl` and `pet_test_manifest.jsonl`: one line per stored corruption (type, severity, settings, seed). Open one in Notepad to see what "stored corruption settings" looks like.
- `configs/*.yaml`: every setting in plain text, including the Optuna search ranges.
- `models/*.onnx` (what the app runs) and `checkpoints/*.pt` (PyTorch weights): you cannot read them, but their sizes are listed in `show_results.py`.

## 6. Test it the way the evaluator will

The evaluator clones the repository into an empty folder and starts it. Do exactly that once, **after the models are published as a GitHub Release**:

```powershell
cd $HOME\Desktop
git clone https://github.com/aleenababar04/genai-assignment-1.git eval-test
cd eval-test
python scripts\download_models.py          # downloads the 7 .onnx files into models\
docker compose up --build                  # first start takes a few minutes
```

(Run `docker compose down` in your original project folder first so port 8080 is free.) Open http://localhost:8080 and repeat parts 3.1 to 3.4 quickly. Then delete the `eval-test` folder.

## 7. Sign-off checklist

Before submitting, everything here should be ticked.

- [ ] `pytest` (184 passed) and `pytest backend/tests` (73 passed)
- [ ] App starts from a fresh clone with one command; **7 of 7 models loaded**
- [ ] All four workspaces tested with your own images, including a download from each
- [ ] Error cases (PDF upload, oversized file, backend stopped) handled without a crash
- [ ] You opened every figure in the gallery and can explain it
- [ ] `python scripts\show_results.py` agrees with the tables in the report
- [ ] W&B project opens and shows trials, final runs and eval runs
- [ ] GitHub repository has the latest code; the Release with the 7 ONNX files exists; the README instructions work
- [ ] Report compiles on Overleaf; name, roll number, W&B link, repository link and YouTube link filled in; no red `TODO` boxes left
- [ ] Demo video (5 to 7 minutes) uploaded to YouTube; only the link is in the report
- [ ] AI-use appendix reviewed: every "pending" row either done or reworded honestly
