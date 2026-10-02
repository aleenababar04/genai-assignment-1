# Generative AI: Assignment 1

TODO: student name, roll number, course and semester.

## Project overview

This repository contains four tasks. Task 1 is a universal denoising autoencoder trained on Oxford-IIIT Pet images (128x128) that restores images corrupted at runtime by salt-and-pepper noise, Gaussian blur or rectangular occlusion, and leaves clean images unchanged; it is trained with the loss `alpha * L1 + (1 - alpha) * (1 - SSIM)`. Task 2 replaces the single model with a 4-class corruption classifier and three specialist autoencoders using hard routing, compares oracle routing with predicted routing, and passes clean images through unchanged. Task 3 is a soft mixture-of-experts: the gate is initialised from the classifier and the experts from the specialists, followed by a warm-up phase and joint fine-tuning with a temperature softmax and a balance regulariser. Task 4 is a style-conditioned face-to-sketch conditional GAN trained on FS2K, with a U-Net generator, a PatchGAN discriminator and a learned embedding for the three sketch styles. All tasks use Optuna for hyperparameter search, Weights & Biases for experiment tracking, and ONNX export verified against PyTorch. The models are served by one FastAPI backend and one React + Tailwind frontend with four workspaces, started with a single Docker Compose command.

## Repository layout

```
.
├── configs/            YAML configuration files for training and search
├── src/
│   ├── data/           Datasets, runtime corruptions, data loaders
│   ├── models/         Autoencoders, classifier, mixture-of-experts, GAN
│   ├── training/       Training and Optuna search scripts
│   ├── evaluation/     Metrics and evaluation scripts
│   ├── export/         ONNX export and PyTorch-vs-ONNX verification
│   └── utils/          Seeding, config loading, W&B tracking helpers
├── optuna_studies/     SQLite databases of the Optuna studies
├── manifests/          Dataset split manifests
├── notebooks/          Kaggle launcher instructions and notebooks
├── backend/            FastAPI inference service (own requirements.txt)
├── frontend/           React + Tailwind app with the four workspaces
├── models/             Exported model files used by the backend
├── report/             IEEE-format LaTeX report; figures in report/figures/
├── docs/               Decision log and AI use log
├── tests/              Tests
├── requirements.txt    Training dependencies
├── .env.example        Template for the environment variables
└── README.md
```

TODO: update this tree once the final file layout is settled (including `docker-compose.yml`).

## Quick start with Docker

Requires Docker with Docker Compose.

1. Clone the repository.

   ```
   git clone TODO: repository URL
   cd TODO: repository folder name
   ```

2. Obtain the model files and place them in `models/`. See [Model downloads](#model-downloads).

   TODO: exact file names expected in `models/`.

3. Build and start everything.

   ```
   docker compose up --build
   ```

4. Open the frontend in a browser.

   TODO: frontend URL and port.
   TODO: backend URL and port (and API docs path).

The frontend has four workspaces: "Universal Restoration", "Hard-Routed Restoration", "Soft Mixture-of-Experts Restoration" and "Face-to-Sketch Generator".

TODO: say whether a `.env` file is needed to run the Docker setup.

## Local development setup

These commands are for Windows PowerShell with Python 3.13. The local install uses CPU-only PyTorch; it is meant for development, tests and inference, not for training.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then open `.env` and fill in `WANDB_API_KEY`. The `.env` file is git-ignored.

If PowerShell refuses to run the activation script, allow scripts for the current session first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

TODO: how to run the backend and frontend locally without Docker.
TODO: how to run the tests.

## Training on Kaggle

Training was run on Kaggle GPU notebooks. Step-by-step instructions are in [notebooks/kaggle_launcher.md](notebooks/kaggle_launcher.md).

## Reproducing each task

### Task 1: Universal Restoration

TODO: dataset preparation, config file, search command, training command, evaluation command, expected outputs.

### Task 2: Hard-Routed Restoration

TODO: classifier training, specialist training, oracle versus predicted routing evaluation.

### Task 3: Soft Mixture-of-Experts Restoration

TODO: initialisation from Task 2 checkpoints, warm-up, joint fine-tuning, evaluation.

### Task 4: Face-to-Sketch Generator

TODO: FS2K preparation, config file, training command, evaluation command.

## ONNX export and verification

TODO: export command for each model, verification command, tolerance used, and where the verification output is recorded.

## Experiment tracking

All runs are logged to the Weights & Biases project `genai-a1`.

TODO: W&B project link.

The Optuna studies are stored as SQLite files in `optuna_studies/`.

TODO: list of study files and which task each belongs to.

## Model downloads

TODO: where the trained model files are hosted, download link, and where to place them.

## Report and demo video

- Report: TODO: link or path to the PDF (LaTeX source is in `report/`).
- Demo video: TODO: link.

## Acknowledgements and AI use

AI tools were used during this assignment. Every use is recorded in [docs/ai_use_log.md](docs/ai_use_log.md), which is also included as the AI-use appendix of the report. Design decisions and the evidence behind them are recorded in [docs/decision_log.md](docs/decision_log.md).

TODO: dataset acknowledgements (Oxford-IIIT Pet, FS2K) and any other credits.
