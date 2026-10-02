# Decision log

The assignment requires every technical decision to be justified with the alternatives that were considered and the evidence behind the choice. This file is where that evidence is collected while the work is happening, so the report can be written from it later.

## How to use it

- Add an entry at the moment a decision is made, not at the end. Numbers and reasons are easy to forget.
- One entry per decision. If a decision is later reversed, add a new entry that refers to the old one; do not delete the old one.
- "Sources consulted" means things actually read: papers, documentation pages, lecture slides. Give enough detail to cite them in the report (title, authors, year, URL).
- "Experiment and numbers" means a result that exists somewhere checkable: a W&B run name or URL, an Optuna study and trial number, or a table in a notebook. If no experiment was run, write "none" and say why the decision was made without one.
- Anything marked `TODO:` is unfinished. Search for `TODO:` before writing the report.

## Entry template

Copy this block for each new decision.

```markdown
### <short title>

- **Date:** YYYY-MM-DD
- **Decision:** <what was decided, one sentence>
- **Question:** <the question that had to be answered>
- **Alternatives considered:**
  - <option A>: <one line on what it is and its main trade-off>
  - <option B>: ...
- **Sources consulted:** <papers, docs, lecture material actually read>
- **Experiment and numbers:** <what was run, where the result is stored, the numbers>
- **Choice and why:** <the option chosen and the reasoning, tied to the evidence above>
- **Report section it feeds:** <section of the IEEE report where this is written up>
```

## Decisions made

### Experiment tracker: Weights & Biases

- **Date:** 2026-10-02
- **Decision:** Use Weights & Biases (project `genai-a1`) for experiment tracking in all four tasks.
- **Question:** Which experiment tracker should record training runs, given that training does not happen on the local machine?
- **Alternatives considered:**
  - Weights & Biases: hosted service; runs are uploaded as training proceeds and stay available online after the Kaggle session ends. Needs an account and an API key stored as a Kaggle secret.
  - MLflow with the local file store: no account needed, but the run data is written to the session's disk on Kaggle and would have to be downloaded and carried back manually after every session.
- **Sources consulted:**
  - Assignment brief (Instructions, third bullet): experiments "must be recorded using either MLflow or Weights & Biases", so both options are allowed.
  - The comparison itself came from the Claude Code session on 2026-10-02 (see `docs/ai_use_log.md`), not from first-hand reading.
  - Not yet read first-hand; read and cite before writing the report: W&B documentation (https://docs.wandb.ai/) and MLflow tracking documentation (https://mlflow.org/docs/latest/).
- **Experiment and numbers:** None. The choice follows from the workflow constraint (training on Kaggle, sessions wiped at the end), not from a measurement, so no comparison run was made.
- **Choice and why:** Weights & Biases. Training runs on Kaggle, and W&B keeps the runs in the cloud, where they survive the end of the session and can be shown directly in the demo video. MLflow's local file store would have to be carried back from Kaggle by hand.
- **Report section it feeds:** Experimental setup, experiment-tracking subsection.

### Training location: Kaggle GPU, CPU-only PyTorch locally

- **Date:** 2026-10-02
- **Decision:** Train on Kaggle GPU notebooks, with Google Colab as the backup. Install CPU-only PyTorch on the laptop and use it for development, tests and inference only.
- **Question:** Where should the models be trained, given the hardware available?
- **Alternatives considered:**
  - Train locally: the laptop has only Intel Iris Xe integrated graphics, so there is no CUDA GPU and training would run on CPU.
  - Kaggle GPU notebooks: free GPU sessions; needs the code to be cloned into the notebook and outputs to be saved before the session ends.
  - Google Colab: similar to Kaggle; kept as the backup if Kaggle is unavailable or the quota runs out.
- **Sources consulted:**
  - Hardware check run on the laptop on 2026-10-02: Windows reports one video adapter, "Intel(R) Iris(R) Xe Graphics"; `nvidia-smi` is not installed; 15.75 GB RAM; about 25 GB free on C:.
  - Not yet read first-hand; read and cite before writing the report: Kaggle notebooks documentation (https://www.kaggle.com/docs/notebooks) for the current GPU quota and session limits.
- **Experiment and numbers:** No timing experiment yet. The installed build is `torch 2.14.1+cpu`. To add once Task 1 trains: seconds per epoch on the laptop CPU and on the Kaggle GPU for the same config.
- **Choice and why:** Kaggle GPU, with Colab as the backup. The laptop has only Intel Iris Xe graphics, so GPU training is not possible locally. CPU-only PyTorch is enough locally for writing code, running tests and serving the exported models.
- **Report section it feeds:** Experimental setup, hardware and environment subsection.

### Local environment: SQLAlchemy installed without compiled extensions

- **Date:** 2026-10-02
- **Decision:** On the Windows laptop, install SQLAlchemy (the library Optuna uses for SQLite study storage) as pure Python, without its compiled extension files.
- **Question:** How can Optuna's SQLite storage run locally when Windows Smart App Control blocks some compiled files in the virtual environment?
- **Alternatives considered:**
  - Turn Smart App Control off: fixes every package, but lowers the machine's protection and may not be reversible without resetting Windows.
  - Run all Python in a Docker dev container: avoids the Windows policy, but needs Docker Desktop running for every test.
  - Do all Python work on Kaggle: no local change, but slow for small tests.
  - Reinstall SQLAlchemy without compiled extensions: nothing is left for the policy to block; slightly slower, which does not matter for Optuna's bookkeeping.
- **Sources consulted:** The error message itself ("DLL load failed while importing _processors_cy: An Application Control policy has blocked this file") and `Get-MpComputerStatus`, which reported `SmartAppControlState: On`. The reinstall approach came from the Claude Code session (see `docs/ai_use_log.md`).
- **Experiment and numbers:**
  - Before: `onnx`, `scikit-learn` and SQLAlchemy 2.1.2 failed to import; `pytest` gave 3 passed, 1 failed (`test_create_study_is_reloadable`).
  - After adding the `.venv` folder to the Windows Security exclusions: `onnx` 1.23.1 and `scikit-learn` 1.9.1 imported; SQLAlchemy was still blocked. Smart App Control still reported `On`, so it is not certain the exclusion was the cause.
  - After reinstalling SQLAlchemy with `DISABLE_SQLALCHEMY_CEXT=1` and `--no-binary sqlalchemy`: no `.pyd` files left in the package; `pytest` gave 4 passed.
- **Choice and why:** Pure-Python SQLAlchemy. It removed the blocked files without changing a security setting, and the tests confirm Optuna studies can be created and reloaded. The commands are recorded at the top of `requirements.txt`. This affects only the laptop; Kaggle and the Docker containers run Linux.
- **Report section it feeds:** Limitations / implementation difficulties.

### Data: direct resize to 128x128 and a cache of clean images

- **Date:** 2026-10-03
- **Decision:** Convert every Pet image to RGB, resize it directly to 128x128 with bicubic interpolation (no cropping), and store the resized clean images once as one `.npy` array per split (`data/pet_128/`).
- **Question:** How should images of different sizes and shapes be brought to 128x128, and should JPEGs be decoded on every load?
- **Alternatives considered:**
  - Direct resize: keeps the whole image but changes the aspect ratio of non-square images.
  - Resize the short side then centre-crop: keeps the aspect ratio but cuts off the sides, which would also cut part of any image an evaluator uploads to the application.
  - Decode and resize JPEGs in every `__getitem__`: no extra disk use, but repeats the same work in every epoch and every Optuna trial.
  - Cache the resized clean images: about 180 MB per 3,680 images, loaded once into memory.
- **Sources consulted:** Assignment brief, "Dataset for Tasks 1-3": images "should be converted to RGB and resized to 128 x 128 pixels", and corrupted copies must not be saved. The cache holds clean images only, so it does not break that rule.
- **Experiment and numbers:** None yet. To add: time per epoch with and without the cache if this is questioned.
- **Choice and why:** Direct resize plus a clean-image cache. The brief says "resized", nothing is cropped away, the application can apply the identical step to uploads, and the cache removes repeated JPEG decoding without storing any corrupted image.
- **Report section it feeds:** Dataset preparation.

### Data: 80/20 split saved to a committed file

- **Date:** 2026-10-03
- **Decision:** Sort the official trainval names, shuffle them with `numpy.random.default_rng(42)`, take the first 80% as training and the rest as validation, and save the name lists to `manifests/pet_split.json`. The official test list is stored in the same file and not used until final evaluation.
- **Question:** How can the same split be guaranteed across Tasks 1, 2 and 3 and across machines?
- **Alternatives considered:**
  - Recompute the split from the seed in each training script: depends on every script, library version and input order agreeing.
  - Save the split to a file once and have every task read it: the split cannot drift, and `prepare_pet.py` raises an error if a recomputed split ever differs from the saved one.
- **Sources consulted:** Assignment brief: "Divide it into 80% training and 20% validation data using random seed 42 ... The same data split must be used throughout Tasks 1, 2, and 3."
- **Experiment and numbers:** Unit tests in `tests/test_prepare_pet.py`: 3,680 names give 2,944 train and 736 validation, the two lists are disjoint, and the result does not depend on input order. On the real dataset (2026-10-03): 2,944 train, 736 validation, 3,669 test images; cache arrays of shape (N, 128, 128, 3).
- **Choice and why:** Saved split file, because it is the only option that makes "the same split" checkable.
- **Report section it feeds:** Dataset preparation.

### Data: corruption design (sampling separated from applying; non-overlapping occlusion)

- **Date:** 2026-10-03
- **Decision:** Implement each corruption as a sampler that returns plain parameters and an apply function that takes them. Salt-and-pepper acts on whole pixels (all three channels together). Occlusion rectangles are placed so they do not overlap, and a sample is accepted only if the measured coverage is inside 10-35% (training) or within 0.01 of the target (test).
- **Question:** How can one implementation serve random training corruption, fixed validation/test corruption and the application, and how can "jointly cover between 10% and 35%" be guaranteed?
- **Alternatives considered:**
  - One function that samples and applies together: simpler, but the parameters cannot be stored or replayed.
  - Separate sample and apply steps: parameters can be written to a manifest and replayed exactly.
  - Occlusion with overlapping rectangles and rejection on the union area: valid, but many samples are rejected at high coverage.
  - Occlusion with non-overlapping rectangles: the union equals the sum of areas, so the target coverage is hit almost always on the first attempt.
  - Salt-and-pepper per channel: produces coloured specks, not black or white pixels as the brief describes.
- **Sources consulted:** Assignment brief, corruption table and fixed test severities. `torchvision.transforms.functional.gaussian_blur` is used for the blur; its documentation is still to be read first-hand (padding mode and kernel definition matter for matching it in the backend).
- **Experiment and numbers:** `tests/test_corruptions.py`, 25 tests passed. Occlusion sampler over 2,000 draws on 128x128: mean attempts 1.0 (low), 1.0 (medium), 1.104 (high, max 4), 1.0185 (training ranges, max 2).
- **Choice and why:** Separate sample/apply with non-overlapping occlusion. It makes the manifests possible and the coverage rule exact, at almost no rejection cost.
- **Report section it feeds:** Dataset preparation, corruption configuration.

### Data: validation and test manifests

- **Date:** 2026-10-03
- **Decision:** Test manifest: 10 entries per test image (clean, plus salt-and-pepper, blur and occlusion at the three fixed severities). Validation manifest: 4 entries per validation image (one per input condition), with severities drawn from the training ranges using a stored per-entry seed. Both are JSON Lines files in `manifests/`, generated once and committed.
- **Question:** What should the deterministic validation set contain?
- **Alternatives considered:**
  - One random condition per validation image (736 items): matches the training distribution, but each condition is measured on about 184 images, so the Optuna objective is noisier and class balance is only approximate.
  - All four conditions per validation image (2,944 items): exactly balanced, and each condition is measured on all 736 images.
  - Using the three fixed test severities for validation too: would tune hyperparameters on exactly the test settings; drawing from the training ranges keeps validation closer to training.
- **Sources consulted:** Assignment brief: validation and test corruptions "must be deterministic"; the manifest must store "the corruption type, severity, mask coordinates, blur settings, and random seed".
- **Experiment and numbers:** `tests/test_manifests.py`, 7 tests passed: manifests are identical across two builds, the fixed severities match the brief, and a manifest dataset returns bit-identical corrupted images on repeated loads, including salt-and-pepper noise. Generated manifests (2026-10-03): 2,944 validation entries and 36,690 test entries (3,669 per condition/severity). Measured test occlusion coverage: 0.0976-0.1025 (low, 1 rectangle), 0.1956-0.2042 (medium, 2), 0.3439-0.3566 (high, 3). Preview grid: `report/figures/corruption_preview.png`.
- **Choice and why:** All four conditions per validation image, for a balanced and less noisy validation objective at negligible cost.
- **Report section it feeds:** Dataset preparation; Optuna search design (validation objective).

## Upcoming decisions

Empty headings for decisions that still have to be made. Fill each one in with the template above when the decision is taken.

### Task 1: Universal denoising autoencoder

#### Bottleneck design

TODO:

#### Skip connections (whether to use them, and where)

TODO:

#### SSIM implementation and window size

TODO:

#### Loss weight alpha between L1 and (1 - SSIM)

TODO:

### Task 2: Classifier and hard-routed specialists

#### Classifier architecture

TODO:

#### Specialist architecture and shared versus separate hyperparameter search

TODO:

#### Identity bypass for clean inputs

TODO:

### Task 3: Soft mixture-of-experts

#### Warm-up length and joint fine-tuning schedule

TODO:

#### Gate temperature

TODO:

#### Balance regulariser (form and weight)

TODO:

### Task 4: Style-conditioned face-to-sketch GAN

#### How the style embedding is injected into the generator

TODO:

#### How the style embedding is injected into the discriminator

TODO:

#### GAN loss type and reconstruction loss weight

TODO:

#### PatchGAN receptive field

TODO:

### Cross-cutting

#### Optuna search spaces, sampler, pruner and trial budget

TODO:

#### ONNX opset version and verification tolerance

TODO:

#### How trained models are distributed to the evaluator

TODO:

#### Frontend and backend structure (one app, four workspaces)

TODO:
