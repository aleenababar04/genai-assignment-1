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
