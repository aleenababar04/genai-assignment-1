# Running training on Kaggle

The laptop is CPU-only, so training runs in a Kaggle notebook with a GPU. This page lists the one-time setup and the cells to paste into the notebook.

Names in angle brackets, such as `<your-username>/<repo>`, are placeholders to replace.

## One-time setup

1. Create a new Kaggle notebook.
2. In the notebook's settings panel:
   - set the accelerator to a GPU;
   - turn internet access on.

   Both options require a phone-verified Kaggle account. If they are missing or greyed out, verify the phone number in the Kaggle account settings first.
3. Add the Weights & Biases key as a secret: **Add-ons > Secrets**, add a secret with the label `WANDB_API_KEY` and the key as its value, and attach it to this notebook.
4. If the GitHub repository is private, add a second secret (for example `GITHUB_TOKEN`) holding a GitHub personal access token that can read the repository.

Secrets have to be attached to each new notebook that uses them.

## Storage: what survives a session

`/kaggle/working` is wiped between sessions unless the output is saved (for example by using "Save Version", or by downloading files before the session ends). Do not rely on it for checkpoints. Log checkpoints to W&B as artifacts so they survive, and copy the Optuna databases out at the end (last cell below).

## Cells to paste

### 1. Read the W&B key and export it

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "genai-a1"
```

### 2. Clone the repository

Public repository:

```python
!git clone https://github.com/<your-username>/<repo>.git
%cd <repo>
```

Private repository (uses the token stored as the `GITHUB_TOKEN` secret):

```python
token = secrets.get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/<your-username>/<repo>.git
%cd <repo>
```

Never paste the token itself into a cell. If git prints the clone URL (for example in an error message), clear that cell's output before saving or sharing the notebook, because the URL contains the token.

### 3. Install dependencies

```python
!pip install -r requirements.txt
```

PyTorch is already installed on Kaggle with GPU support. If `requirements.txt` lists torch, pip will see the requirement as already satisfied and keep the installed version. If pip does start downloading a different torch build, stop and check the version pin, because replacing Kaggle's build can break GPU support.

### 4. Check the GPU

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

This must print `True`. If it prints `False`, the accelerator is not enabled in the notebook settings.

### 5. Run a training script

```python
!python -m src.training.<script> --config configs/<file>.yaml
```

Run it from the repository root (the `%cd <repo>` in cell 2 takes care of that). Replace `<script>` and `<file>` with the training module and config for the task being run.

### 6. Copy the Optuna studies out for download

```python
!cp optuna_studies/*.db /kaggle/working/
```

The repository was cloned into `/kaggle/working/<repo>`, so this places the `.db` files at the top of the output folder where they are easy to find. Download them from the notebook's output panel, put them in `optuna_studies/` in the local copy of the repository, and commit them.

## Before closing the session

- Confirm the run appears in the W&B project `genai-a1`, with its checkpoints uploaded as artifacts.
- Download the Optuna `.db` files (cell 6).
- Anything else left only in `/kaggle/working` will be lost unless the notebook version is saved.

## Resuming an Optuna study in a later session

A new session starts with a fresh clone, so it only has the study databases that were committed. Commit and push the downloaded `.db` files before starting the next session if the search should continue from where it stopped.
