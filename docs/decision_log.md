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
- **Sources consulted:** TODO: list the documentation pages or other material actually read for this comparison.
- **Experiment and numbers:** TODO: none run yet. If a comparison is made (for example, logging the same short run to both), record it here; otherwise write "none" and explain.
- **Choice and why:** Weights & Biases. Training runs on Kaggle, and W&B keeps the runs in the cloud, where they survive the end of the session and can be shown directly in the demo video. MLflow's local file store would have to be carried back from Kaggle by hand.
- **Report section it feeds:** TODO: experimental setup / experiment tracking subsection.

### Training location: Kaggle GPU, CPU-only PyTorch locally

- **Date:** 2026-10-02
- **Decision:** Train on Kaggle GPU notebooks, with Google Colab as the backup. Install CPU-only PyTorch on the laptop and use it for development, tests and inference only.
- **Question:** Where should the models be trained, given the hardware available?
- **Alternatives considered:**
  - Train locally: the laptop has only Intel Iris Xe integrated graphics, so there is no CUDA GPU and training would run on CPU.
  - Kaggle GPU notebooks: free GPU sessions; needs the code to be cloned into the notebook and outputs to be saved before the session ends.
  - Google Colab: similar to Kaggle; kept as the backup if Kaggle is unavailable or the quota runs out.
- **Sources consulted:** TODO: list anything actually read (for example Kaggle or Colab documentation on GPU quotas and session limits).
- **Experiment and numbers:** TODO: none run yet. If epoch time is measured on CPU and on the Kaggle GPU, record both here.
- **Choice and why:** Kaggle GPU, with Colab as the backup. The laptop has only Intel Iris Xe graphics, so GPU training is not possible locally. CPU-only PyTorch is enough locally for writing code, running tests and serving the exported models.
- **Report section it feeds:** TODO: experimental setup / hardware and environment subsection.

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

#### Corruption parameters and sampling at runtime

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
