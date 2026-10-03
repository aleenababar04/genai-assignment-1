# Generative AI: Assignment 1

TODO: student name, roll number, course and semester.

Four generative image systems, trained in PyTorch, exported to ONNX and served through one web application:

| Task | System | Workspace in the app |
|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder | Universal Restoration |
| 2 | Corruption classifier + three specialist autoencoders with hard routing | Hard-Routed Restoration |
| 3 | Jointly trained soft mixture-of-experts (gate from Task 2's classifier, experts from its specialists) | Soft Mixture-of-Experts Restoration |
| 4 | Style-conditioned face-to-sketch conditional GAN (U-Net generator, PatchGAN discriminator) | Face-to-Sketch Generator |

Tasks 1-3 use the Oxford-IIIT Pet dataset (128 x 128) with salt-and-pepper noise, Gaussian blur and rectangular occlusion applied at runtime. Task 4 uses FS2K. Every task is tuned with Optuna and tracked in Weights & Biases.

## Run the application (Docker)

Requirements: Git and Docker Desktop (or Docker Engine with the Compose plugin). Nothing else needs to be installed.

1. **Clone the repository**

   ```
   git clone https://github.com/aleenababar04/genai-assignment-1.git
   cd genai-assignment-1
   ```

2. **Get the model files.** The seven ONNX files are attached to the GitHub Release [`models-v1`](https://github.com/aleenababar04/genai-assignment-1/releases/tag/models-v1). Either run

   ```
   python scripts/download_models.py
   ```

   (any Python 3.8+, no packages needed), or download the seven `.onnx` files from the release page by hand and put them in the `models/` folder:

   ```
   models/task1_udae.onnx
   models/task2_classifier.onnx
   models/task2_specialist_salt_pepper.onnx
   models/task2_specialist_blur.onnx
   models/task2_specialist_occlusion.onnx
   models/task3_moe.onnx
   models/task4_generator.onnx
   ```

3. **Start everything with one command**

   ```
   docker compose up --build
   ```

4. **Open http://localhost:8080** in a browser.

The sidebar's *System status* card shows whether the backend is online and which models are loaded. If a model file is missing, the app still starts; that workspace shows a "model not loaded" message. No `.env` file is needed to run the application. To stop: `Ctrl+C`, then `docker compose down`.

How it fits together: the `frontend` container (nginx) serves the React app and forwards every `/api/...` request to the `backend` container (FastAPI + ONNX Runtime, no PyTorch). `models/` is mounted read-only into the backend.

### Backend API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Backend status, which ONNX models are loaded, last inference time |
| `GET /api/samples`, `GET /api/samples/{id}` | Sample images for the gallery |
| `POST /api/restore/universal` | Task 1 |
| `POST /api/restore/hard-routing` | Task 2: classifier probabilities, predicted corruption, selected expert |
| `POST /api/restore/soft-moe` | Task 3: four routing weights |
| `POST /api/sketch` | Task 4 |

The restore endpoints take a multipart form with `file` (PNG/JPG, max 10 MB) or `sample_id`, plus `corruption` (`none`, `salt_pepper`, `blur`, `occlusion`), `severity` (`low`, `medium`, `high`) and an optional `seed`. `/api/sketch` takes `file` or `sample_id` and `style` (1, 2 or 3). Interactive API documentation is served by FastAPI at `/docs` on the backend (uncomment the backend `ports` line in `docker-compose.yml` to reach it at http://localhost:8000/docs).

## Repository layout

```
.
├── src/
│   ├── data/          Pet and FS2K preparation, runtime corruptions, manifests, datasets
│   ├── models/        Autoencoder, classifier, hard router, soft MoE, conditional GAN
│   ├── training/      Training + Optuna scripts (one per model)
│   ├── evaluation/    Test-set evaluation, metrics, figures, routing analysis
│   ├── export/        ONNX export and PyTorch-vs-ONNX verification
│   └── utils/         Seeding, config loading, W&B and Optuna helpers
├── configs/           One YAML file per task (+ *_best.yaml written by Optuna)
├── manifests/         Fixed data splits and the validation/test corruption manifests
├── optuna_studies/    Optuna studies (SQLite, one file per study)
├── notebooks/         Kaggle notebooks that launch the training scripts on a GPU
├── backend/           FastAPI inference service, its Dockerfile and tests
├── frontend/          React + Tailwind CSS application, its Dockerfile and nginx config
├── scripts/           download_models.py
├── models/            ONNX files (downloaded, not committed)
├── report/            IEEE LaTeX report, figures and result tables
├── docs/              Decision log, AI-use log, Google Stitch prompts
├── tests/             Tests for the training code
├── docker-compose.yml
└── requirements.txt   Training / evaluation dependencies
```

## Reproducing the training

Training was run on Kaggle GPU notebooks; the notebooks in `notebooks/` contain only the commands below plus data download and result collection. Run the commands from the repository root.

**Environment** (local development on Windows; on Linux use `python3 -m venv` and `source .venv/bin/activate`):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build; Kaggle already has a GPU build
pip install -r requirements.txt
Copy-Item .env.example .env    # then put your WANDB_API_KEY in .env
```

**Data**

```
python -m src.data.prepare_pet      # Oxford-IIIT Pet: download, 80/20 split (seed 42), 128x128 cache
python -m src.data.manifests        # fixed validation and test corruption manifests (already committed)
python -m src.data.prepare_fs2k     # FS2K: needs the dataset in data/FS2K (see the Task 4 notebook)
```

**Training, evaluation and export** (each `--mode optuna` writes `configs/<task>_best.yaml`, which `--mode final` then uses):

| Task | Commands | Notebook |
|---|---|---|
| 1 | `python -m src.training.train_udae --mode optuna` <br> `python -m src.training.train_udae --mode final` <br> `python -m src.training.train_udae --mode final --skip` (skip-connection ablation) <br> `python -m src.evaluation.eval_udae` <br> `python -m src.export.export_onnx --task task1` | `kaggle_task1_udae.ipynb` |
| 2 | `python -m src.training.train_classifier --mode optuna` / `--mode final` <br> `python -m src.training.train_specialists --mode optuna` / `--mode final` <br> `python -m src.evaluation.eval_task2` <br> `python -m src.export.export_onnx --task task2` | `kaggle_task2_hard_routing.ipynb` |
| 3 | needs the Task 2 checkpoints in `checkpoints/` <br> `python -m src.training.train_moe --mode optuna` / `--mode final` <br> `python -m src.evaluation.eval_moe` <br> `python -m src.export.export_onnx --task task3` | `kaggle_task3_moe.ipynb` |
| 4 | `python -m src.training.train_cgan --mode optuna` / `--mode final` <br> `python -m src.evaluation.eval_cgan` <br> `python -m src.export.export_onnx --task task4` | `kaggle_task4_cgan.ipynb` |

Evaluation writes result tables to `report/results/` and figures to `report/figures/`. The export step checks every ONNX model against PyTorch on the same inputs (maximum absolute difference must be at most 1e-4) and writes the measured difference to `report/results/<model>_onnx_check.json`.

**Tests**

```
python -m pytest                    # training code
python -m pytest backend/tests      # backend (needs: pip install -r backend/requirements.txt -r backend/requirements-dev.txt)
```

**Running the app without Docker** (for development):

```
python -m uvicorn app.main:app --app-dir backend --port 8000
cd frontend && npm install && npm run dev      # then open http://localhost:5173
```

## Experiment tracking

All training runs, Optuna trials, evaluation results and checkpoints are logged to the Weights & Biases project `genai-a1`: TODO: project link. The Optuna studies are in `optuna_studies/` (`task1_udae.db`, `task2_classifier.db`, `task2_specialists.db`, `task3_moe.db`, `task4_cgan.db`) and can be summarised with `python scripts/show_results.py` (or opened interactively after `pip install optuna-dashboard`: `optuna-dashboard sqlite:///optuna_studies/<file>.db`).

## Documentation

| File | Purpose |
|---|---|
| [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) | What the assignment is and how every part was built and explained, for exam preparation |
| [docs/testing_and_outputs_guide.md](docs/testing_and_outputs_guide.md) | How to test the application, and where every output is and what it means |
| [docs/decision_log.md](docs/decision_log.md) | Every design decision with the alternatives and the evidence |
| [docs/ai_use_log.md](docs/ai_use_log.md) | Every use of AI tools and how the output was checked |
| `scripts/show_results.py` | Prints all key results in one summary |
| `scripts/build_gallery.py` | Builds `report/outputs_gallery.html`, a page showing every figure with an explanation |
| `scripts/demo_temperature.py` | Shows how the Task 3 temperature changes the routing weights |

## Report and demo video

- Report: TODO: path to the PDF (LaTeX source in `report/`).
- Demo video: TODO: YouTube link.

## Acknowledgements and AI use

- Oxford-IIIT Pet Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, "Cats and Dogs", CVPR 2012 (CC BY-SA 4.0).
- FS2K: D.-P. Fan et al., FS2K dataset, https://github.com/DengPingFan/FS2K. TODO: full citation.
- The generator and discriminator follow pix2pix: P. Isola, J.-Y. Zhu, T. Zhou and A. A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017.

AI tools were used during this assignment. Every use is recorded in [docs/ai_use_log.md](docs/ai_use_log.md), which is also the AI-use appendix of the report. Design decisions and the evidence behind them are in [docs/decision_log.md](docs/decision_log.md).
