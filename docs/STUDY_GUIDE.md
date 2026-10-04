# Study guide: Generative AI, Assignment 1

Written for you, to read before the evaluation. It explains what the assignment is, what we built, how and why we built it, and how to talk about it. Every number in it comes from files in this repository (run `python scripts/show_results.py` to see them all at once).

**How to use it.** Read Parts 1 to 3 first (about 40 minutes): they give you the vocabulary. Then Parts 4 to 7, one task at a time (about 20 minutes each). Part 12 is a question-and-answer rehearsal for the evaluation, Part 13 is a "how to change things" cheat sheet, and Part 14 covers the live demo and what to do if something breaks. Read the code files listed in Part 16 yourself: the evaluator may ask you to explain or modify any of them.

## Contents

1. [What the assignment asks](#part-1-what-the-assignment-asks)
2. [Concepts you need](#part-2-concepts-you-need)
3. [The data](#part-3-the-data)
4. [Task 1: universal denoising autoencoder](#part-4-task-1-universal-denoising-autoencoder)
5. [Task 2: classifier, specialists and hard routing](#part-5-task-2-classifier-specialists-and-hard-routing)
6. [Task 3: soft mixture-of-experts](#part-6-task-3-soft-mixture-of-experts)
7. [Task 4: face-to-sketch conditional GAN](#part-7-task-4-face-to-sketch-conditional-gan)
8. [How the work was done, step by step](#part-8-how-the-work-was-done-step-by-step)
9. [The application and the deployment](#part-9-the-application-and-the-deployment)
10. [Numbers to remember](#part-10-numbers-to-remember)
11. [Limitations (say them before they are asked)](#part-11-limitations)
12. [Evaluation questions and answers](#part-12-evaluation-questions-and-answers)
13. [Cheat sheet: changing things during the evaluation](#part-13-cheat-sheet-changing-things)
14. [Live demo and troubleshooting](#part-14-live-demo-and-troubleshooting)
15. [Where everything is](#part-15-where-everything-is)
16. [Your homework before the evaluation](#part-16-your-homework-before-the-evaluation)

---

## Part 1: What the assignment asks

### 1.1 In one paragraph

Build four small generative-AI systems, train and tune them properly, then put all four inside **one web application** that anyone can start with one command. The first three systems **repair damaged photos of cats and dogs**. The fourth **draws a pencil sketch from a face photo** in one of three styles. You must also document every decision with evidence, in an IEEE-format research-style report, and record a demo video.

### 1.2 The four tasks

| Task | What it does | In the app |
|---|---|---|
| **1. Universal denoising autoencoder** | One neural network repairs any of three kinds of damage (noise, blur, black boxes) without being told which one it is dealing with | *Universal Restoration* |
| **2. Corruption classifier + specialist autoencoders** | A classifier first guesses the kind of damage, then sends the image to one specialist network that only knows how to repair that kind. Clean images skip repair ("identity bypass") | *Hard-Routed Restoration* |
| **3. Soft mixture-of-experts** | Instead of picking one specialist, a "gate" gives each of four branches (identity + three specialists) a weight, and the output is the weighted blend. Gate and specialists are fine-tuned together | *Soft Mixture-of-Experts Restoration* |
| **4. Conditional GAN, face to sketch** | Turns a face photo into a sketch; you choose Style 1, 2 or 3 | *Face-to-Sketch Generator* |

The damage ("corruptions") for Tasks 1 to 3 is created by code, because the dataset contains only clean photos: **salt-and-pepper noise** (random black and white dots), **Gaussian blur**, and **rectangular occlusion** (black boxes covering 10 to 35% of the image).

### 1.3 Everything the brief requires, and where it is

| Requirement | Where it is in the project | Status |
|---|---|---|
| Four models in PyTorch | `src/models/`, `src/training/` | Done |
| Optuna hyperparameter search in each task | `src/training/train_*.py`, results in `optuna_studies/*.db` (5 studies) | Done |
| Experiment tracking (MLflow or W&B) | W&B project `genai-a1` | Done |
| Pet data: official trainval split 80/20 with seed 42, test set untouched | `manifests/pet_split.json`, `src/data/prepare_pet.py` | Done |
| Corruptions created at runtime in the data loader; fixed (deterministic) validation and test manifests | `src/data/corruptions.py`, `pet_dataset.py`, `manifests.py` | Done |
| Task 1: bottleneck autoencoder, loss `alpha*L1 + (1-alpha)*(1-SSIM)`, results per corruption and severity, 12 examples with error maps, 4 failure cases | `report/results/task1_*`, `report/figures/task1_*` | Done |
| Task 2: balanced classifier batches, accuracy / macro P-R-F1 / confusion matrix, three specialists, oracle vs predicted routing, misrouting analysis | `report/results/task2_*`, `report/figures/task2_*` | Done |
| Task 3: gate from classifier, experts from specialists, warm-up then joint training, balance loss, weights per corruption and severity, heatmap, inactive/dominating expert check | `report/results/task3_*`, `report/figures/task3_*` | Done |
| Task 4: FS2K official split + 15% stratified validation, U-Net generator, PatchGAN discriminator, style embedding in both, separate loss logging, fixed validation samples | `src/models/cgan.py`, `src/training/train_cgan.py`, `report/*/task4_*` | Done |
| Export to ONNX and verify against PyTorch | `src/export/export_onnx.py`, `report/results/*_onnx_check.json` | Done (files are in `models/`, not in git) |
| Interface designed in Google Stitch, evidence in report | `report/figures/stitch/` (9 screens + HTML) | Done |
| React + Tailwind frontend, FastAPI backend, four workspaces in one app | `frontend/`, `backend/` | Done |
| Frontend and backend in Docker; one-command Compose start | `docker-compose.yml` | Done, tested locally |
| GitHub repo with code, configs, Optuna studies, ONNX code, Dockerfiles, README; models by link or LFS | GitHub repo; models via a GitHub Release | Release still to be created |
| IEEE-format LaTeX report | `report/main.tex` | Written; needs compile check on Overleaf, your name, final TODOs |
| 5 to 7 minute demo video on YouTube | `docs/demo_video_script.md` is the shot list | To record |
| AI-use appendix | `docs/ai_use_log.md` (also in the report) | Done; some "pending" items are yours to confirm |
| Submit via Google Classroom | | To do |

**About the date.** The PDF header says "Deadline: March 16, 2024". Confirm the real deadline with your instructor.

### 1.4 How you will be evaluated

The brief says you may be asked to:

- justify your architecture and a research-informed decision;
- interpret an experimental result;
- modify part of the implementation;
- run the system on **previously unseen images**, choosing corruptions and severities, uploading an already-corrupted image, inspecting classifier and mixture weights, generating sketches, and **restarting the application from the repository**.

So the evaluation is partly a viva (questions) and partly a live demo. Parts 12 to 14 prepare you for both.

---

## Part 2: Concepts you need

Short and plain. If a word in a later part is unfamiliar, it is here.

### Images and damage

- An image here is **128 x 128 pixels with 3 colour channels** (red, green, blue). That is 128 x 128 x 3 = **49,152 numbers**, each between 0 and 1.
- **Salt-and-pepper noise**: a fraction of pixels is replaced by pure white or pure black.
- **Gaussian blur**: each pixel is replaced by a weighted average of its neighbours. Two settings: the *kernel size* (how many neighbours) and *sigma* (how strongly).
- **Occlusion**: black rectangles cover part of the image.
- **Restoration** means producing the original clean image from the damaged one.

### Neural-network basics

- A **neural network** is a long chain of simple calculations with adjustable numbers called **parameters** (or weights). Our networks have between 0.3 and 42 million of them.
- A **convolution layer** slides a small filter over the image to detect patterns such as edges. **ReLU** is a simple on/off-style function between layers. **BatchNorm** keeps the numbers inside a layer well-scaled so training is stable. **Dropout** randomly switches parts of a layer off during training so the network does not memorise.
- **Training** = show the network many examples, measure how wrong the answer is with a **loss** number, and nudge the parameters to make the loss smaller. The nudging method is **gradient descent**; we use the optimiser **Adam** (or AdamW).
- **Learning rate**: the size of each nudge. Too big: training is unstable. Too small: it is slow. A **cosine schedule** lowers it smoothly to near zero over training.
- **Batch**: how many images are processed at once. **Epoch**: one pass over the whole training set.
- **Validation set**: images the network does not train on, used to pick settings and to decide which epoch was best. **Test set**: touched only at the end, to report honest final results. Using the test set to make choices would be cheating, because the numbers would no longer be honest.
- **Overfitting**: the network memorises the training images and does worse on new ones. Visible when training loss keeps falling but validation gets worse. We did not see this.

### Autoencoders

- An **autoencoder** has an **encoder** (squeezes the image into a small summary), a **bottleneck** (the summary itself, called the *latent*), and a **decoder** (rebuilds an image from the summary).
- A **denoising** autoencoder is given a *damaged* image and trained to output the *clean* one. Because everything must pass through the small bottleneck, noise cannot get through; the network learns what real photos look like.
- The price: fine detail is lost too, so outputs look slightly smooth.
- A **skip connection** is a shortcut that lets some information bypass the bottleneck. The brief forbids unrestricted skips (the network could just copy its input) but allows limited ones if you investigate them. We built one as an experiment (ablation).

### Measuring quality

| Measure | What it is | How to read it |
|---|---|---|
| **L1** | Average absolute difference between output and clean image (0 to 1) | Lower is better. 0.04 means about 4% error per pixel |
| **SSIM** | Structural similarity: compares local contrast, edges and structure (0 to 1) | Higher is better. 1 = identical. 0.8 is good, 0.5 is poor |
| **PSNR** | A log-scale version of the squared error, in decibels (dB) | Higher is better. Roughly: 15 dB bad, 20 dB poor, 25 dB acceptable, 30 dB good, 40 dB very close. Identical images give infinity (we cap it at 100) |

Our training loss is `alpha * L1 + (1 - alpha) * (1 - SSIM)`: L1 makes pixel values accurate; SSIM keeps edges and structure. `alpha` is the balance between them.

### Classification measures

- A classifier outputs raw scores (**logits**), one per class. **Softmax** turns them into probabilities that add up to 1. **Cross-entropy** is the loss that rewards a high probability for the right class.
- **Accuracy**: fraction right. **Precision** of a class: of the images called that class, how many really were. **Recall**: of the images that really were that class, how many were found. **F1**: a combined score of the two. **Macro** average: the plain average over the four classes, so rare classes count equally.
- A **confusion matrix** shows, for each true class (row), how often each answer (column) was given. A perfect classifier has 1.0 on the diagonal.

### Mixture of experts

- Several specialist networks ("experts") plus a **gate** that decides how much to trust each.
- **Hard routing**: pick exactly one expert (argmax). Simple, but a wrong pick is all-or-nothing, and the argmax cannot be trained by gradient descent.
- **Soft routing**: the gate gives each branch a weight, `w = softmax(G(x) / tau)`; the output is the **weighted sum** of all branches. This is smooth, so the gate and experts can be trained together.
- **Temperature `tau`**: small `tau` makes the weights nearly one-hot (almost hard routing); large `tau` spreads them out. Our trained value: 0.85.
- **Routing collapse**: the gate sends everything to one expert, leaving the others unused. A **balance loss** discourages it.

### GANs

- A **GAN** has two networks in a contest. The **generator** makes fake outputs; the **discriminator** tries to tell real from fake. Each improves by beating the other.
- Ours is a **conditional** GAN (pix2pix style): the generator gets a photo (and a style) and the discriminator sees the photo together with a real or generated sketch. The generator's loss is *fool the discriminator* **plus** *L1 distance to the true sketch* (weighted by `lambda_L1`, here about 121). The L1 term keeps the sketch faithful to the photo; the adversarial term gives it realistic strokes.
- **U-Net**: an encoder-decoder with skip connections at every level, so detail from the photo reaches the output. (Unlike Task 1, skips are wanted here.)
- **PatchGAN**: the discriminator judges many small patches (70 x 70 pixels) instead of the whole image, which pushes the generator to get local texture right.
- A **learned style embedding** is a small table of numbers (one row per style) that the network learns during training; the style number picks the row, and the row is fed into the network as extra input.

### Tools

- **Optuna** searches for good hyperparameters automatically. One **trial** = one short training run with one combination of settings. Optuna's **TPE** sampler learns which regions of the settings space work well and tries those more. A **pruner** stops a trial early if it is clearly doing worse than earlier trials at the same epoch (we use the *median pruner*).
- **Weights & Biases (W&B)** records every run (settings, loss curves, sample images, checkpoints) online.
- **ONNX** is a portable file format for trained networks. We train in PyTorch, export to ONNX, and the web backend runs the ONNX files with **ONNX Runtime**, which needs no PyTorch. Export must not change the results, so we check that PyTorch and ONNX outputs agree (largest difference must be below 0.0001; ours are about 0.000001 or smaller).
- **Docker** runs programs in isolated "containers" with everything they need. **Docker Compose** starts several containers together from one file. **FastAPI** is the Python web framework of the backend; **React** and **Tailwind CSS** build the user interface; **nginx** is the web server that serves the interface and forwards API calls to the backend.
- **Kaggle** gave us a free GPU. Training a network on a CPU would take far too long; the notebooks in `notebooks/` just download the code from GitHub and run our training scripts.

---

## Part 3: The data

### 3.1 Oxford-IIIT Pet (Tasks 1 to 3)

- 37 breeds of cats and dogs. We use the **official trainval list (3,680 images)** as development data and keep the **official test list (3,669 images)** untouched until the end.
- The 3,680 are split **80 / 20 with random seed 42** into **2,944 training** and **736 validation** images. The split is saved in `manifests/pet_split.json`, so Tasks 1, 2 and 3 provably use the same split.
- Every image is converted to RGB and resized **directly** to 128 x 128 (no cropping, so the whole picture is kept). The clean, resized images are cached once as `.npy` files (`data/pet_128/`). Only clean images are cached: corrupted copies are never saved.

### 3.2 Corruptions during training (new every time an image is loaded)

Each time the data loader fetches a training image, it picks one of four conditions with probability 1/4 each, then draws fresh random settings:

| Condition | Setting drawn each time |
|---|---|
| Clean | none |
| Salt-and-pepper | probability uniform in 0.02 to 0.15; chosen pixels become black or white (50/50) |
| Gaussian blur | kernel size 3, 5 or 7; sigma uniform in 0.5 to 2.5 |
| Occlusion | 1 to 3 non-overlapping black rectangles covering 10 to 35% of the image in total |

**Why at runtime?** The network sees a new damaged version every epoch (like unlimited training data), and we never store thousands of corrupted copies. The code is in `src/data/corruptions.py`: functions that *sample* settings and functions that *apply* them are separate on purpose, so the same code can replay stored settings.

### 3.3 Fixed corruptions for validation and testing (the "manifests")

Validation and test must be **identical on every run**, so the corruption settings are generated once and stored.

- **Validation manifest** (`manifests/pet_val_manifest.jsonl`): 4 entries per validation image (clean + each corruption with settings drawn from the training ranges, each with its own stored random seed) = **2,944 entries**.
- **Test manifest** (`manifests/pet_test_manifest.jsonl`): 10 entries per test image = **36,690 entries**: clean, plus each corruption at three fixed severities:

| Corruption | Low | Medium | High |
|---|---|---|---|
| Salt-and-pepper probability | 0.03 | 0.08 | 0.15 |
| Blur (kernel, sigma) | (3, 0.7) | (5, 1.5) | (7, 2.5) |
| Occlusion (rectangles, coverage) | 1 box, 10% | 2 boxes, 20% | 3 boxes, 35% |

Each entry stores the type, severity, blur settings, rectangle coordinates and seed. We measured the occlusion coverage in the test manifest: 9.8 to 10.3% (low), 19.6 to 20.4% (medium), 34.4 to 35.7% (high). The preview image `report/figures/corruption_preview.png` shows all ten versions of four test images.

### 3.4 FS2K (Task 4)

- 2,104 pairs of a face photo and a hand-drawn sketch, in three sketch styles. The dataset's own split: **1,058 official training pairs and 1,046 official test pairs**.
- We hold out **15% of the training pairs per style as validation (seed 42)**: **899 train, 159 validation**, and the **1,046 test** pairs are used only for the final evaluation.
- The test set is unbalanced: Style 1 has 619 pairs, Style 2 has 381, Style 3 only 46. Remember this when discussing per-style results.
- Original images come in four sizes; photo and sketch of a pair always have the same size. We resize both to 128 x 128. All sketches turned out to be greyscale, but we keep them as 3-channel images.
- **Augmentation** (random flip, and resize to 143 x 143 then a random 128 x 128 crop) is applied **identically to the photo and its sketch**: both are stacked into one tensor and transformed with one set of random numbers. If they were transformed separately, the pixel-by-pixel correspondence would be destroyed and the GAN would learn nonsense. A unit test checks this.

### 3.5 Balanced batches (Tasks 2 and 3)

The classifier must not favour any class, so every training batch contains **exactly** 25% of each of the four conditions (labels assigned 0,1,2,3,0,1,2,3,... then shuffled, then each image corrupted accordingly). Drawing randomly would be balanced only on average. Code: `src/data/balanced.py`.

---

## Part 4: Task 1, universal denoising autoencoder

### 4.1 The idea

One network, never told which damage it receives, must output the clean image. It learns one shared notion of "what a clean photo looks like". Question the task asks: is that enough?

### 4.2 The architecture (`src/models/autoencoder.py`)

With our chosen settings (`base_channels c = 64`, `latent_channels = 128`):

```
corrupted image  3 x 128 x 128
  stem                     -> 64 channels at 128 x 128
  down block 1 (stride 2)  -> 64  at 64 x 64
  down block 2             -> 128 at 32 x 32
  down block 3             -> 256 at 16 x 16
  down block 4             -> 512 at  8 x  8
  1x1 convolution          -> BOTTLENECK: 128 channels at 8 x 8 = 8,192 numbers
  decoder: 4 x (upsample x2 + 2 convolutions) -> back to 128 x 128
  final convolution + sigmoid -> restored image 3 x 128 x 128, values in [0, 1]
```

- The bottleneck holds **8,192 numbers instead of 49,152**: six times smaller. That is the "meaningful bottleneck" the brief asks for. The decoder sees *only* this bottleneck.
- The encoder halves the width and height four times while multiplying the channels, as the brief describes ("progressively reduce the spatial dimensions while increasing the number of channels").
- The decoder uses *nearest-neighbour upsampling followed by a convolution* instead of transposed convolutions, which avoid the checkerboard artefacts that transposed convolutions are known for (Odena et al., 2016).
- **Dropout2d (0.035)** on the bottleneck switches off whole channels during training as a mild regulariser.
- About **7.8 million parameters**; the ONNX file is 31 MB.

### 4.3 Loss and training (`src/training/losses.py`, `train_udae.py`)

- Loss = `alpha * L1 + (1 - alpha) * (1 - SSIM)`, with **alpha = 0.552** chosen by Optuna (the brief suggested starting at 0.8 and not accepting it blindly).
- Adam, learning rate 6.3e-4 with cosine decay, batch size 32, **80 epochs** (about 43 minutes on the Kaggle GPU). Each epoch: train on freshly corrupted batches, then score on the fixed validation manifest; the epoch with the best validation objective is kept.

### 4.4 The Optuna search

- **What was searched** (as required): learning rate, batch size, bottleneck channels, encoder channels, dropout, and alpha.
- **How trials were scored**: every trial uses the *same* fixed validation objective, `0.8 * L1 + 0.2 * (1 - SSIM)`, measured on the validation manifest. Why fixed? Different trials train with different alphas, so their own training losses cannot be compared; Optuna would just prefer whichever alpha makes the number small.
- **Budget**: 30 trials of 10 epochs; **19 completed, 11 pruned**. Best: trial 20.
- **Chosen**: lr 6.3e-4, batch 32, bottleneck 128, base channels 64, dropout 0.035, alpha 0.552.
- **What mattered** (fANOVA importances): learning rate 0.67, dropout 0.21, alpha 0.10; bottleneck size, base channels and batch size hardly mattered in the tested ranges.
- Figure: `report/figures/task1_udae_optuna.png`.

### 4.5 Results (test set; each row is 3,669 images)

| Input | PSNR before -> after | SSIM before -> after |
|---|---|---|
| Salt-and-pepper, high | 13.1 -> 25.1 dB | 0.20 -> 0.79 |
| Blur, high | 24.4 -> 24.7 dB | 0.69 -> 0.75 |
| Occlusion, high | 10.7 -> 20.0 dB | 0.54 -> 0.65 |
| **All corrupted images (33,021)** | **19.3 -> 24.0 dB** | **0.635 -> 0.758** |
| Clean | infinite -> 25.2 dB | 1.00 -> 0.80 |
| Blur, low | 32.3 -> 25.4 dB | 0.94 -> 0.80 |

**How to explain it:** the network's output quality is almost the same (about 25 dB) whatever the input. That level is a ceiling set by the bottleneck. So heavily damaged images improve a lot, but clean and mildly blurred images, which were *better* than 25 dB, get **worse**. Noise is removed almost completely; occlusion is improved but the hidden content is only guessed (blurry colour blends); blur is hardly improved because blur already removed the detail.

### 4.6 The skip-connection ablation

We retrained the same configuration with **one** skip connection that passes the 16 x 16 encoder features to the decoder (`--skip`).

| | No skip (our model) | With skip |
|---|---|---|
| Corrupted images, PSNR / SSIM | 23.96 dB / 0.758 | 26.77 dB / 0.852 |
| Clean images, PSNR | 25.21 dB | 29.96 dB |

The skip helps everywhere, most where detail exists in the input (+4.8 dB clean, +4.2 dB noise, +3.0 dB blur) and least for occlusion (+1.2 dB, because the missing content is simply not there to pass along). **Why we keep the no-skip model:** it satisfies the "genuine bottleneck" requirement without any argument. A 16 x 16 skip is still eight times smaller than the image, so it does not let the network copy its input (high noise is still restored from 13 to about 29 dB), but defending that is optional work we avoided. The skip model is in the report as the investigation the brief asks for.

### 4.7 Failure cases (important for the evaluation)

`report/figures/task1_udae_failures.png`: the four worst test images, one per condition.

- A **cat on a pure black background** is the worst case for clean and occlusion. The network has learned that large black areas are occlusion masks, so it "repairs" the black background by painting it grey-brown, even when the input was perfectly clean. The network is blind to *which* corruption it faces, and black boxes are exactly what occlusion looks like. **This is a key finding: it motivates Task 2** (identify the corruption explicitly).
- A **dog in a dense flower field** is the worst for noise and blur: the dense fine texture cannot pass through the bottleneck, so the output is a smeared, painterly version.

### 4.8 Where it lives

Model `src/models/autoencoder.py`; loss `src/training/losses.py`; training and Optuna `src/training/train_udae.py`; config `configs/task1_udae.yaml` and the Optuna result `configs/task1_udae_best.yaml`; evaluation `src/evaluation/eval_udae.py`; export `src/export/export_onnx.py`; Kaggle launcher `notebooks/kaggle_task1_udae.ipynb`; in the app: *Universal Restoration*.

---

## Part 5: Task 2, classifier, specialists and hard routing

### 5.1 The idea

Instead of one model for everything: (1) a **classifier** says which corruption is present; (2) a **specialist** autoencoder that was trained *only* on that corruption repairs it; (3) if the image is clean, **nothing is done** (identity bypass, the output is the input).

```
image -> classifier -> probabilities [clean, salt, blur, occlusion] -> argmax
   clean      -> output = input (no expert is run)
   salt       -> salt-and-pepper specialist
   blur       -> blur specialist
   occlusion  -> occlusion specialist
```

### 5.2 The classifier (`src/models/classifier.py`)

Four blocks of (convolution, BatchNorm, ReLU) x 2 + max-pooling, then global average pooling, dropout and a linear layer to **4 scores**. The first block works at full 128 x 128 resolution on purpose: what separates the classes (single noisy pixels, slight softness) is fine detail that early downsampling would erase. Optuna chose the **smallest** size preset (0.29 M parameters). The output is raw logits; softmax is applied afterwards (and the Task 3 gate reuses this exact network).

Training: cross-entropy loss, AdamW, **balanced batches** (exactly 25% per class, see 3.5), 25 epochs. Optuna tuned learning rate, batch size, channel preset, dropout and weight decay, **maximising validation macro-F1**: 12 trials of 6 epochs (**3 completed, 9 pruned**). Chosen: lr 3.6e-4, batch 32, small, dropout 0.43, weight decay 6.4e-5.

### 5.3 Classifier results (test set, 36,690 images)

- **Accuracy 0.9961** (143 wrong in 36,690); **macro precision 0.9922, macro recall 0.9950, macro F1 0.9936**.
- Salt-and-pepper: perfect (1.000). Blur: 0.9998 recall. Clean: precision 0.972 (some corrupted images are called clean). Occlusion: recall 0.9906.
- All medium and high severities are classified with **100% accuracy**.
- **The hardest case**: *low occlusion* (one small black box). 101 of those 3,669 images (2.8%) are called clean. We had predicted clean vs. low blur would be hardest; the data refuted that (only 2 low-blur images were called clean). Say so: it shows you tested a hypothesis.
- Figure: `report/figures/task2_test_confusion.png` (the confusion matrix, nearly the identity matrix).

### 5.4 The three specialists (`src/training/train_specialists.py`)

Each specialist is the **same architecture as Task 1** but trained *only* on its own corruption (salt-and-pepper only, blur only, occlusion only), with its own independent parameters (about 7.5 M each). The brief allows a **shared Optuna search** to find one architecture, then training the three independently:

- In each trial, the three specialists are trained side by side and the trial's score is the mean of their validation objectives. 10 trials of 5 epochs, **3 completed, 7 pruned**. Chosen: lr 4.7e-4, batch 32, bottleneck 64, base channels 64, alpha 0.864. Then each specialist is trained alone for 40 epochs with its own W&B run.

### 5.5 Oracle vs predicted routing (`src/models/hard_router.py`)

The brief wants two test modes:

- **Oracle routing**: the *true* corruption label from the test manifest chooses the expert. This measures how good the specialists are.
- **Predicted routing**: the classifier chooses. This measures the real, deployable system.

The router runs an expert **only on the images routed to it**, and returns clean-routed images **bit-for-bit unchanged** (tests confirm this).

### 5.6 Results and what they mean

| (corrupted images only) | PSNR | SSIM |
|---|---|---|
| Input | 19.25 dB | 0.635 |
| Task 1 universal | 23.96 dB | 0.758 |
| Task 2 oracle routing | 23.63 dB | 0.705 |
| Task 2 predicted routing | 23.63 dB | 0.705 |

- **Predicted routing is practically identical to oracle routing**, because the classifier is so accurate.
- **The big win over Task 1 is clean images**: Task 1 degraded them to 25.2 dB; hard routing returns them unchanged in 99% of cases.
- **Honest finding:** on corrupted images the specialists are **not better** than the universal model (23.63 vs 23.96 dB). Specialisation alone does not remove the bottleneck ceiling of about 25 dB. (The comparison is not perfectly fair: specialists had a smaller search, 40 vs 80 epochs, and a bottleneck of 64 vs 128 channels.)

### 5.7 Misrouting analysis (required by the brief)

143 of 36,690 images were sent to the wrong branch (`report/results/task2_misrouting.csv`):

| True class | Sent to | Images | Effect |
|---|---|---|---|
| Occlusion (low) | identity | 101 | Harmless, even **better** (+0.17 SSIM: the occlusion expert softens the whole image to fix a small box) |
| Clean | blur | 22 | Costly: SSIM drops by 0.17 |
| Clean | occlusion | 16 | Costly: SSIM drops by 0.37 |
| Blur | identity | 2 | Better, +0.31 SSIM |
| Occlusion | blur | 2 | Slightly worse |

`report/figures/task2_misrouted.png` shows the six worst: **all are clean photos with large black areas** (black background, black letterbox bars, a black warning sign). The classifier reads them as occlusion, and the occlusion expert "repairs" them. Same root cause as Task 1's failure, now at the routing decision.

### 5.8 Where it lives

`src/models/classifier.py`, `hard_router.py`, `src/data/balanced.py`, `src/training/train_classifier.py`, `train_specialists.py`, `src/evaluation/eval_task2.py`, `classification.py`; configs `task2_classifier.yaml`, `task2_specialists.yaml` and their `_best.yaml`; notebook `kaggle_task2_hard_routing.ipynb`; four ONNX files (classifier + three specialists). In the app: *Hard-Routed Restoration* (the backend does the routing: it runs the classifier ONNX, takes the argmax and runs the chosen specialist ONNX).

---

## Part 6: Task 3, soft mixture-of-experts

### 6.1 The idea

Hard routing makes an all-or-nothing choice, which fails for ambiguous or mixed cases. Task 3 makes the choice **soft and trainable**:

```
w = softmax( G(x) / tau )            G = the gate network, w = [w0, w1, w2, w3]
output = w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occlusion(x)
```

`x` itself is the identity branch. All four weights are positive and add up to 1. Because the weighted sum is smooth, the **gate and the experts can be trained together** from the final reconstruction error.

### 6.2 Starting point and two-stage training (`src/training/train_moe.py`)

- The brief forbids starting from random weights. The **gate is the trained Task 2 classifier**; the **experts are the three trained Task 2 specialists**.
- **Stage 1, warm-up (2 epochs):** experts frozen, only the gate trains (lr 5e-4). Why: the gate's scores must first become meaningful as *weights* without wrecking the good experts.
- **Stage 2, joint fine-tuning (15 epochs):** everything unfrozen, smaller learning rate (8e-5, cosine decay).
- **Detail worth knowing:** a "frozen" expert still changes if BatchNorm keeps updating its running statistics in training mode. So frozen experts are also kept in *evaluation mode* (an override of `train()` in `SoftMoERestorer`). A test checks that frozen experts keep identical BatchNorm statistics.

### 6.3 The loss

`L = l1 * L1 + ls * (1 - SSIM) + lc * CrossEntropy + lb * Balance`

- **L1 and (1 - SSIM)**: reconstruction, as before (`l1 = 0.705`, `ls = 0.295`).
- **Cross-entropy** (`lc = 0.038`): keeps the gate related to the known corruption label. It is computed on `logits / tau`, i.e. on the same distribution that produces the weights.
- **Balance** (`lb = 0.0054`): `sum over k of (mean weight of branch k - 1/4)^2`, over a balanced batch. It is zero when each branch gets 25% on average and grows if the gate favours some branches, which prevents routing collapse.
- Temperature chosen: **tau = 0.85**.

### 6.4 The Optuna search

Learning rate, tau, cross-entropy weight, balance weight and the L1/SSIM weighting. 8 trials of (1 warm-up + 4 fine-tune) epochs, trial 0 fixed to the brief's starting values. **All 8 completed; none was pruned for routing collapse** (a trial would be stopped if any branch's average weight fell below 0.05; they stayed near 0.2). The importances are spread evenly (0.16 to 0.25 each). Figure `task3_moe_optuna.png`.

### 6.5 Results

Table in the report and `report/results/task3_moe_test_metrics.csv`. On corrupted images: **23.71 dB / SSIM 0.750**, close to Task 1 and better than Task 2 in SSIM. On clean images: **62.4 dB / SSIM 0.999**, i.e. almost untouched (not exactly 1.0 because the identity weight is 0.98, not 1). On **mild** corruptions it is the best system, for example low blur: **27.5 dB** vs 25.4 (Task 1) and 24.7 (Task 2).

### 6.6 What the gate learned (the core of this task)

Average weights on the test set (`report/results/task3_moe_mean_weights.csv`, heatmap `task3_moe_routing_heatmap.png`):

| True input | Identity | Salt-pepper | Blur | Occlusion |
|---|---|---|---|---|
| Clean | **0.98** | 0.00 | 0.01 | 0.01 |
| Salt-and-pepper, high | 0.00 | **1.00** | 0.00 | 0.00 |
| Blur, low | 0.42 | 0.00 | **0.58** | 0.00 |
| Blur, high | 0.05 | 0.00 | **0.95** | 0.00 |
| Occlusion, low | **0.62** | 0.00 | 0.00 | 0.38 |
| Occlusion, high | 0.01 | 0.00 | 0.00 | **0.99** |

Read this as three facts:

1. Every corruption goes to **its own expert**, and the routing gets **sharper as the corruption gets stronger**.
2. The weight not given to the matching expert goes to the **identity branch**, never to a wrong expert. For mild corruption the gate hedges: "mostly repair, but keep some of the original." It learned this on its own, matching Task 2's finding that leaving a mild corruption alone can beat sending it to an expert.
3. **Expert health** (`task3_moe_expert_health.csv`): every expert is the top choice for about 22 to 30% of images; none is inactive; none dominates unrelated inputs. The identity branch's weight of 0.155 on other corruptions is the deliberate hedging, not domination.

Why is the gate's agreement with the true label (about 0.95) lower than the classifier's (0.99)? Because it is trained to minimise reconstruction error, not to name the corruption; Optuna picked a small cross-entropy weight, which allows this.

Examples: `task3_moe_dominant_examples.png` (one expert wins), `task3_moe_distributed_examples.png` (weights shared: mild or ambiguous cases such as a low occlusion on a black cat). Failures: the same two images as before; the weights in the row labels separate **routing errors** (clean cat sent to the occlusion expert, 0.87) from **expert errors** (correct routing but the expert cannot recover the content).

### 6.7 Cost

All three experts run for every image: about 120 ms per image in the CPU backend against about 40 ms for Task 1. The whole pipeline (gate, experts, mixing, and the temperature as a constant) is exported as **one ONNX file** (91 MB) that returns the image and the four weights.

### 6.8 Where it lives

`src/models/soft_moe.py`, `src/training/train_moe.py`, `src/evaluation/eval_moe.py`, `routing_analysis.py`; `configs/task3_moe.yaml`; notebook `kaggle_task3_moe.ipynb`; in the app: *Soft Mixture-of-Experts Restoration*. Live demo of the temperature: `python scripts/demo_temperature.py`.

---

## Part 7: Task 4, face-to-sketch conditional GAN

### 7.1 The idea

Learn to turn a face photo into a sketch in a chosen style (Style 1, 2 or 3), using the 899 training pairs of FS2K.

### 7.2 The networks (`src/models/cgan.py`)

- **Generator, a U-Net** (about 42 M parameters): the photo goes down seven times (128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1) and comes back up, with **skip connections at every level** so detail from the photo reaches the sketch. Dropout in the innermost up-layers; tanh output.
- **Discriminator, a PatchGAN** (2.8 M parameters): outputs a 14 x 14 map of real/fake scores, each looking at a 70 x 70 patch of a (photo, sketch) pair.
- **Style condition, built into BOTH networks** (the brief insists it must not be just an interface label): each network owns a separate learned `Embedding(3, 32)`. The generator gets the style embedding **twice** (concatenated to the photo at the input, and to the 1 x 1 bottleneck); the discriminator gets its own embedding as an extra input map alongside the photo and sketch. Our style-swap figure proves the generator really uses it.

### 7.3 Losses and training (`src/training/train_cgan.py`)

- Discriminator: `0.5 * (BCE(real pair -> 1) + BCE(generated pair -> 0))`.
- Generator: `BCE(generated pair -> 1)  +  lambda_L1 * L1(true sketch, generated sketch)`.
- Each step: update the discriminator on a real and a detached fake pair, then the generator.
- **Logged separately every epoch** (as required): discriminator real loss, discriminator fake loss, generator adversarial loss, generator L1 loss, and validation L1/PSNR/SSIM overall and per style. The **same validation photographs** are logged as sample grids every 10 epochs, so you can watch the generator improve.
- Adam with betas (0.5, 0.999); final run **200 epochs** (about 70 minutes; the brief allows shorter Optuna trials followed by a full retraining). A first final run of 100 epochs was replaced by the 200-epoch run because its **validation** objective was lower (0.3030 vs 0.3073); on the test set the two are practically identical (SSIM 0.473 vs 0.472), i.e. the GAN had converged by about epoch 100.

### 7.4 The Optuna search

Both learning rates, batch size, base channels, dropout, style-embedding size and `lambda_L1`. 8 trials of 15 epochs, **3 completed, 5 pruned**. Chosen: generator lr 1.2e-4, discriminator lr 4.5e-4, batch 4, base channels 64, dropout 0.30, style embedding 32, **lambda_L1 = 121** (near pix2pix's default 100). Importances: discriminator lr 0.35, dropout 0.33, generator lr 0.24, lambda_L1 0.08.

**How were trials scored?** GAN losses do not measure quality (they oscillate by design). FID, the usual GAN metric, is unreliable with only about 160 validation images. FS2K is *paired*, so each trial is scored against the true sketches with `0.5 * L1 + 0.5 * (1 - SSIM)`.

### 7.5 Results (official test split)

| Group | Pairs | L1 | PSNR | SSIM |
|---|---|---|---|---|
| All | 1,046 | 0.108 | 15.4 dB | 0.473 |
| Style 1 | 619 | 0.080 | 17.1 dB | 0.519 |
| Style 2 | 381 | 0.156 | 12.2 dB | 0.381 |
| Style 3 | 46 | 0.069 | 18.2 dB | 0.608 |

**How to explain it:** Style 2 is much harder to score because it is a dense, high-contrast drawing with many individual hair strokes. A plausible stroke pattern in the "wrong" place is penalised pixel by pixel. Style 3's number rests on only 46 test images. Pixel metrics understate the visual quality of dense styles.

- `task4_cgan_style_swap.png`: the same photo in all three styles: Style 1 light outline, Style 2 heavy dark shading, Style 3 soft toned sketch; identity and pose preserved. **This is the figure that proves the conditioning works.**
- `task4_cgan_failures.png`: the four worst test pairs are all Style 2 portraits of people with long, voluminous hair (SSIM 0.18 to 0.19): recognisable and in the right style, but strokes placed differently from the artist's.
- Training curves (`task4_cgan_curves.png`): the discriminator's two losses fall together and the generator's adversarial loss rises slowly, so the discriminator gradually gains the upper hand, which is normal; nothing collapsed or diverged; validation SSIM rose from 0.41 to about 0.48 in the first 100 epochs and then stayed flat while the training L1 kept falling: the generator had converged, which is why 200 epochs gave the same test result as 100.

### 7.6 Where it lives

`src/models/cgan.py`, `src/data/prepare_fs2k.py`, `fs2k_dataset.py`, `src/training/train_cgan.py`, `src/evaluation/eval_cgan.py`; `configs/task4_cgan.yaml`; notebook `kaggle_task4_cgan.ipynb`; one ONNX file with **only the generator** (168 MB; the discriminator is a training tool only). In the app: *Face-to-Sketch Generator* (photo in [0,1] plus a style number 1 to 3 in the app and API; the backend subtracts 1 because the model uses 0 to 2 inside).

---

## Part 8: How the work was done, step by step

This is the story to tell if you are asked "how did you approach it?".

1. **Understood the brief** and made an eight-phase plan: setup, data, Tasks 1 to 4, ONNX, application, report and video.
2. **Setup.** GitHub repo, Python environment (CPU only on your laptop), accounts (Kaggle for GPU, W&B for tracking), a fixed folder layout, one YAML config per task, fixed random seeds, and two running logs: `docs/decision_log.md` (every decision, the alternatives, the evidence) and `docs/ai_use_log.md` (every use of AI and how it was checked). The brief demands both.
3. **Data pipeline** before any model: download Pet, fixed split, corruption functions, manifests, datasets, balanced batches, all with automatic tests (the corruption tests check things like "occlusion coverage is within the allowed range" and "the same seed gives the same noise").
4. **For each task**: write the model, loss, metrics, a training script that can run both an Optuna search and a final training, an evaluation script that writes tables and figures, an ONNX export with a verification step, and tests. Everything was smoke-tested for a few batches on the CPU first, so the expensive GPU run would not fail on a typo.
5. **Training on Kaggle.** Each notebook clones the repo, downloads the data, runs the Optuna search (every trial is a W&B run), then the final training, then evaluation and ONNX export, and zips the outputs. You downloaded the zips and put the results into the repo.
6. **Reading the results critically** and writing them into the report: tables, figures, failure analysis, interpretations (this is what the brief calls "analysis").
7. **Time budget.** The deadline forced smaller Optuna searches and shorter final trainings for Tasks 2 to 4 than first planned (recorded in the decision log as a limitation).
8. **The application.** FastAPI backend (loads ONNX files, validates uploads, applies corruptions, runs inference), designed the interface in **Google Stitch** first, built the React + Tailwind frontend, restyled it to match the Stitch screens, wrapped both in Docker with Compose, and tested the containers.
9. **Documentation.** README with run instructions, the IEEE report, this guide, the demo plan.
10. **Remaining:** publish models as a GitHub Release, test from a fresh clone, record the video, compile the report on Overleaf, submit.

**Reproducibility** (a good thing to point out): the seed 42 is set everywhere, the splits and the validation/test corruptions are stored files, every run is logged, every choice is in the decision log, and `pytest` runs 184 + 73 tests.

---

## Part 9: The application and the deployment

### 9.1 What the user sees

One web page at **http://localhost:8080** with a sidebar of four workspaces (plus an Overview) and a status card (backend online, models loaded, last inference time). The three restoration workspaces share the same input panel: upload a PNG/JPG or pick a sample, choose a corruption (none, salt-and-pepper, blur, occlusion) and a severity (low, medium, high), press Restore. Results show three images side by side (the original, the corrupted image the model receives, and the restored output), the corruption settings, the inference time and a download button. The hard-routed page also shows the four classifier probabilities, the predicted corruption and the chosen expert (or "Identity bypass"); the soft page shows the four routing weights as a stacked bar with the top contributor highlighted. The face page accepts an upload or a webcam capture, a style choice, and shows photo and sketch side by side with a download button.

### 9.2 What happens when you press Restore (the request flow)

```
Browser (React)  --POST /api/restore/hard-routing (photo + settings)-->  nginx  -->  FastAPI backend
   1. validate: type PNG/JPG, size <= 10 MB, really an image
   2. resize to 128 x 128 RGB (same as in training)
   3. apply the chosen corruption (the backend has its own NumPy/OpenCV copy of the corruption code,
      checked against the PyTorch version: blur matches to 3e-7)
   4. run the ONNX model(s) with ONNX Runtime
         universal: one model  |  hard routing: classifier -> argmax -> chosen specialist (none if clean)
         soft: one model returning image + 4 weights  |  sketch: generator with (photo, style)
   5. encode result images as PNG, measure the time
Browser  <-- JSON { input_image, output_image, corruption settings, probabilities / weights, inference_ms } --
```

### 9.3 The pieces

- **Backend** (`backend/app/`): `main.py` (endpoints), `models.py` (loads the 7 ONNX files at startup and reports which loaded), `inference.py` (the four operations), `corruptions.py`, `imaging.py`, `config.py`. Endpoints: `GET /api/health`, `GET /api/samples`, `POST /api/restore/universal`, `/hard-routing`, `/soft-moe`, `POST /api/sketch`. It does not need PyTorch.
- **Frontend** (`frontend/src/`): React 19 + Tailwind CSS 4 built with Vite. Pages in `pages/`, reusable pieces in `components/`, API calls in `api.js`. Page navigation is a URL hash (`#/universal` and so on). It shows real data only (placeholders in the Stitch mock-ups were replaced by real backend values).
- **Stitch design** (`report/figures/stitch/`): the interface was designed first in Google Stitch (prompts in `docs/stitch_prompts.md`), then implemented to match, as the brief requires.
- **Docker** (`docker-compose.yml`): two services. `backend` (Python 3.12 image, reads the `models/` folder through a read-only volume, has a health check). `frontend` (built with Node, served by nginx, starts only after the backend is healthy; nginx forwards `/api/` to the backend, so the browser talks to a single address). **One command: `docker compose up --build`**, then open http://localhost:8080.

### 9.4 The deployment requirement

The brief says: "The complete application must run locally through Docker Compose. Public hosting through a suitable service is optional, but a working local containerized deployment is compulsory." That compulsory part is done and tested. The evaluator's path is: clone the repo, run `python scripts/download_models.py` (the ONNX files are too big for git, so they are attached to a GitHub Release), run `docker compose up --build`, open the browser.

**Public hosting is optional and not done.** If asked: it would mean putting the same two containers on a hosting service (for example a container platform, or Hugging Face Spaces), uploading about 380 MB of models (the generator alone is 168 MB), and keeping it online during the assessment. We chose the compulsory local deployment. Say this plainly; do not claim a public URL.

### 9.5 Tests

`python -m pytest` (184 tests: data, corruptions, models, losses, routing, ONNX export) and `python -m pytest backend/tests` (73 tests: API validation, corruption equivalence with PyTorch, endpoints). Plus manual testing with the guide in `docs/testing_and_outputs_guide.md`.

---

## Part 10: Numbers to remember

| Fact | Value |
|---|---|
| Image size | 128 x 128 x 3 = 49,152 numbers |
| Pet split | 2,944 train / 736 validation / 3,669 test (seed 42) |
| Test manifest | 36,690 entries (10 per image); validation manifest 2,944 |
| FS2K split | 899 train / 159 validation / 1,046 test; test styles 619 / 381 / 46 |
| Task 1 bottleneck | 8 x 8 x 128 = 8,192 numbers (6x smaller than the image); alpha 0.552 |
| Task 1 result (corrupted) | 19.25 -> 23.96 dB; SSIM 0.635 -> 0.758; clean images degraded to 25.2 dB |
| Task 1 skip ablation | 26.77 dB / 0.852 (corrupted); clean 29.96 dB |
| Task 1 Optuna | 30 trials: 19 completed, 11 pruned; best trial 20; lr matters most (0.67) |
| Task 2 classifier | accuracy 0.9961, macro-F1 0.9936; 143 errors in 36,690; hardest: low occlusion (101 sent to "clean") |
| Task 2 routing | oracle = predicted = 23.63 dB / 0.705 (corrupted) |
| Task 3 | 23.71 dB / 0.750 (corrupted); clean 62.4 dB / 0.999; tau 0.85; all 8 trials completed; no collapse |
| Task 3 gate | high-severity weights 0.95 to 1.00 on the right expert; low blur 0.42 identity / 0.58 blur |
| Task 4 | SSIM 0.473 overall (Style 1: 0.519, Style 2: 0.381, Style 3: 0.608); lambda_L1 = 121; 200 epochs (100 epochs gave 0.472) |
| Optuna studies | 5: 30, 12, 10, 8, 8 trials |
| Training time | Task 1 final: 43 minutes (80 epochs); trials 2 to 6 minutes each |
| ONNX agreement | largest PyTorch-ONNX difference about 1e-6 or less (limit 1e-4) |
| ONNX file sizes | Task 1 31 MB; classifier 1.2 MB; each specialist 30 MB; Task 3 91 MB; generator 168 MB |
| Tests | 184 + 73 |
| Latency (measured in the Docker backend on your laptop CPU) | Task 1 about 40 ms; hard routing about 90 ms; soft MoE about 120 ms; sketch about 40 ms |

---

## Part 11: Limitations

Say these yourself; they show understanding.

1. **Reduced Optuna budgets and shorter final trainings** (Tasks 2 to 4) to meet the deadline; the searches are coarse. (The GAN was retrained for the full 200 epochs; it had already converged at about 100.)
2. **Task 1's ceiling:** the bottleneck limits detail, so clean and mildly corrupted images are made slightly worse. Black backgrounds are mistaken for occlusions.
3. **Specialists were not better than the universal model** on corrupted images, and the comparison is not perfectly controlled (different search sizes, epochs and bottleneck widths).
4. **The fixed validation objective** (`0.8 * L1 + 0.2 * (1 - SSIM)`) slightly favours trials whose alpha is near 0.8.
5. **GAN evaluation** uses paired L1/SSIM, which understates the quality of dense drawing styles; no FID. Only 46 Style 3 test pairs.
6. **Soft MoE cost:** every image runs all three experts.
7. **Corruptions are synthetic** (the three types only); real-world damage may behave differently.
8. **The app resizes everything to 128 x 128** without cropping, so non-square photos are stretched.
9. **No public hosting;** local Docker only.
10. **Face model only knows faces** from one dataset.

---

## Part 12: Evaluation questions and answers

Short model answers. Use your own words; the point is to understand the reasoning.

### The problem and the data

**1. Why not train one model per corruption, or one model for everything?** We tried both ends: Task 1 is one model for everything (simple, but it cannot tell clean from damaged, so it degrades clean images), Task 2 uses one specialist per corruption chosen by a classifier (clean images pass through unchanged, but errors are all-or-nothing), and Task 3 blends them (the best compromise).

**2. Why generate corruptions at runtime?** The network sees a fresh damaged version of each image every epoch, which acts like unlimited training data, and no corrupted copies need to be stored.

**3. Why fixed manifests for validation and test?** So every model and every run is judged on exactly the same damaged images; results become comparable and reproducible.

**4. Why seed 42, and why must the test set stay untouched?** The seed makes the split reproducible. The test set measures honest performance; if we chose settings by looking at it, the reported numbers would be optimistic.

**5. Why resize to 128 x 128?** The brief specifies it. It also keeps training fast.

### Task 1

**6. What is the bottleneck and how big is it?** The 8 x 8 x 128 tensor between encoder and decoder: 8,192 numbers against 49,152 in the image, six times smaller. The decoder sees only this.

**7. Why no skip connections?** The brief forbids unrestricted skips and requires a genuine bottleneck. We tested one limited skip as an ablation: it improves quality (23.96 to 26.77 dB) because detail bypasses the bottleneck, but we keep the skip-free model so the requirement is satisfied without argument.

**8. Why L1 plus SSIM? What does alpha do?** L1 makes individual pixels accurate; SSIM keeps structure (edges, contrast). `alpha` is the balance; Optuna chose 0.552, giving SSIM more weight than the suggested 0.8.

**9. What do PSNR and SSIM mean, and why does the model make clean images worse?** PSNR is error in decibels (higher is better); SSIM is structural similarity from 0 to 1. The model's output quality is capped near 25 dB by the bottleneck, so inputs better than that (clean, mild blur) get worse and inputs worse than that improve.

**10. What is the main failure and why?** Cats on black backgrounds: black boxes are what occlusion looks like, so the network "repairs" black areas. It cannot know which corruption it faces. This motivated Task 2.

**11. Why Optuna's fixed validation objective?** Trials train with different alphas, so their own training losses are not comparable; a fixed formula on the validation manifest gives one comparable number per trial.

**12. Which hyperparameter mattered most?** The learning rate (importance 0.67 in Task 1), then dropout and alpha.

### Optuna and tracking

**13. What is Optuna, a trial, TPE, pruning?** Automatic hyperparameter search. A trial is a short training run with one setting combination. TPE proposes new settings by learning which regions worked well. The median pruner stops trials that are worse than the median at the same epoch, saving compute.

**14. Why so few trials?** Time. Task 1 had 30 trials; Tasks 2 to 4 had 8 to 12. It is stated as a limitation. The required elements (a study per task over the required hyperparameters, then a longer final training) are all present.

**15. Why W&B instead of MLflow?** Training ran on Kaggle, whose disk is wiped after each session; W&B keeps runs online and shows them in the demo.

### Task 2

**16. Why balanced batches and how?** To stop the classifier being biased towards a class. Labels are assigned by position in the batch (0,1,2,3,0,1,...), shuffled, then each image is corrupted accordingly, so every batch is exactly 25% each.

**17. What did the classifier get wrong?** 143 of 36,690. Mostly low occlusion called clean (101), and clean images with black areas called blur or occlusion (38).

**18. Why oracle and predicted routing?** Oracle (true label) isolates the specialists' ability; predicted shows the deployable system. They are identical here, because the classifier is so accurate.

**19. Why an identity bypass?** A clean image needs no repair, and an autoencoder would only degrade it. The brief requires it.

**20. Why were the specialists not better than Task 1?** They share the same bottleneck ceiling, and they were found with a smaller search and trained for fewer epochs. Specialisation helps with clean images (bypass), not with the detail limit.

### Task 3

**21. Explain softmax with temperature.** Softmax turns scores into positive weights that sum to 1. Dividing the scores by `tau` first controls sharpness: small `tau` nearly one-hot, large `tau` spread out. Run `python scripts/demo_temperature.py` to show it.

**22. Why initialise from Task 2 and warm up with frozen experts?** Random weights would give meaningless mixtures. The warm-up lets the gate adapt to producing weights without damaging the good experts. Then everything is fine-tuned together at a smaller learning rate.

**23. Why keep frozen experts in eval mode?** BatchNorm updates its running statistics in training mode even without gradients, so "frozen" experts would drift.

**24. Explain the four loss terms.** L1 and (1 - SSIM) for reconstruction; cross-entropy to keep the gate tied to the known corruption; balance loss `sum (mean weight - 1/4)^2` to prevent collapse onto one expert.

**25. What is routing collapse and did it happen?** The gate sending everything to one expert. We prune any trial whose smallest average branch weight drops below 0.05. It never happened: all 8 trials completed and each expert is the top choice for about 22 to 30% of images.

**26. Why is the gate less accurate than the classifier?** It optimises reconstruction, not labels. For mild corruptions it shares weight with the identity branch because that reconstructs better; that is deliberate and visible in the weights.

**27. Why is Task 3 one ONNX file but Task 2 four?** Soft mixing has no data-dependent branching, so it exports as a single graph (with `tau` baked in) that returns the image and the four weights. Hard routing branches on the argmax, so the backend does the routing between four separate models.

### Task 4

**28. How does the conditional GAN work?** The generator makes a sketch from photo plus style; the discriminator judges (photo, sketch, style) as real or fake. The generator is trained to fool it and to stay close (L1) to the true sketch.

**29. How is the style used?** Each network has its own learned embedding of the style. The generator gets it at the input and at the bottleneck; the discriminator gets it as an extra input. The style-swap figure shows the effect.

**30. Why a U-Net and a PatchGAN?** The U-Net's skips carry the photo's structure into the sketch. The PatchGAN judges local 70 x 70 patches, which sharpens textures and works with few training images.

**31. Why must augmentation be identical for photo and sketch?** Otherwise the pixel correspondence is destroyed and the model learns from mismatched pairs.

**32. Why not FID?** It needs many images; with about 160 validation images it is biased and noisy. We use paired L1/SSIM against the true sketches.

**33. Why is Style 2 scored lower?** It is dense with many strokes whose exact positions cannot be predicted; pixel metrics penalise plausible but different strokes.

### Engineering

**34. What is ONNX and how did you verify it?** A portable network format, run in the backend with ONNX Runtime and no PyTorch. After exporting, the same inputs go through PyTorch and ONNX; the largest difference must be under 0.0001. Ours are about 1e-6.

**35. Explain the app architecture.** React frontend served by nginx; nginx forwards `/api/` calls to the FastAPI backend, which validates the upload, resizes it, applies the chosen corruption, runs the ONNX model(s) and returns PNGs, probabilities or weights, and timing. Both run in Docker; one command starts them.

**36. Why does the backend have its own corruption code?** The user picks a corruption in the interface and the backend must create it without PyTorch. It was checked against the PyTorch version in tests (the blur matches to 3e-7).

**37. How would you deploy it publicly?** Push the two images to a container host, mount or bake in the models, expose the frontend port, keep it up during assessment. Not done; the brief makes it optional.

**38. How did you make the results reproducible?** Fixed seeds, stored splits and manifests, config files, W&B logging, the decision log, and tests.

**39. What did the AI tools do, and what did you verify?** Be honest and specific. AI assistants wrote much of the code, drafted documents and helped debug; every use is logged in `docs/ai_use_log.md`. You ran the training, designed the interface in Stitch, ran and checked results, and you must be able to explain the code. Say which files you have read and understood (see Part 16). The brief allows AI use but requires that you verify and understand everything.

**40. What would you improve with more time?** Longer Optuna searches, a fairer specialist-versus-universal comparison, a wider bottleneck or a perceptual loss to reduce smoothing, better handling of black backgrounds (for example training with images that have black borders), and public hosting.

---

## Part 13: Cheat sheet, changing things

The evaluator may say "change X". Here is where X lives. **Before any retraining on your laptop, copy the `checkpoints` folder to a backup** (training writes a new checkpoint file in place; `--mode optuna` also adds trials to the `optuna_studies/*.db` file and overwrites `configs/*_best.yaml`, so back those up too), and set `WANDB_MODE=disabled` so no run is uploaded. Full retraining needs the GPU; on the CPU you can only run a tiny demonstration with `--epochs 1 --max-batches 5 --num-workers 0` (for Task 3 use `--warmup-epochs 1 --finetune-epochs 1` instead of `--epochs`).

| Change | Where | Then |
|---|---|---|
| Corruption strength ranges (noise probability, blur kernels and sigma, occlusion coverage) | `src/data/corruptions.py`, the constants at the top (`SP_PROB_RANGE`, `BLUR_SIGMA_RANGE`, `OCC_COVERAGE_RANGE`, `TEST_*`); the app copy is `backend/app/corruptions.py` | Run `pytest`; to change the stored manifests: `python -m src.data.manifests --force` (then every model must be re-evaluated) |
| Loss weight alpha (Task 1) | `alpha` in `configs/task1_udae_best.yaml` | `python -m src.training.train_udae --mode final` (GPU) |
| Bottleneck size | `latent_channels` in the same file | same retraining; the compression ratio is `latent_channels * 64` against 49,152 |
| Number of Optuna trials or epochs | `optuna:` block of `configs/taskN_*.yaml` | `python -m src.training.train_X --mode optuna` |
| Search ranges | `search_space:` block of the same files | same |
| Temperature `tau` (Task 3) | live: `python scripts/demo_temperature.py`; permanent: change `tau` in `configs/task3_moe_best.yaml`, retrain, re-export | `python -m src.export.export_onnx --task task3` |
| Hard-routing rule (for example "only trust the classifier above 80% confidence, otherwise leave the image alone") | `hard_routed` in `backend/app/inference.py` (this changes only the app; the test results in the report come from `HardRoutedRestorer` in `src/models/hard_router.py`) | `docker compose up --build -d backend` |
| Upload size limit | `MAX_UPLOAD_MB` in `backend/app/config.py` | rebuild the backend |
| What the app shows (text, colours, layout) | `frontend/src/constants.js`, `index.css`, `pages/`, `components/` | `docker compose up --build -d frontend` |
| Number of trials per study shown in a table | read it from the database: `python scripts/show_results.py` | |
| A new endpoint | `backend/app/main.py` (add a route calling a function in `inference.py`) | rebuild the backend; `pytest backend/tests` |
| Show a model's parameter count | `python -c "from src.models.autoencoder import ConvAutoencoder, count_parameters; print(count_parameters(ConvAutoencoder(64,128)))"` | |

Useful quick commands to *show* things without changing anything: `python scripts/show_results.py` (all numbers), `python scripts/demo_temperature.py` (routing weights vs tau), `python -m pytest -q` (tests), and the app itself.

---

## Part 14: Live demo and troubleshooting

### 14.1 Starting the app (what the evaluator will do, and what you should rehearse)

```
git clone https://github.com/aleenababar04/genai-assignment-1.git
cd genai-assignment-1
python scripts/download_models.py        # fetches the 7 ONNX files from the GitHub Release
docker compose up --build                # then open http://localhost:8080
```

Check that the status card says **Backend online** and **7 of 7 loaded**. First start-up takes a few minutes (building images). Stop with `Ctrl+C` and `docker compose down`.

### 14.2 A good order for the live demo (about 5 minutes)

1. Overview page, then **Universal Restoration** with the sample cat: salt-and-pepper high, then occlusion medium. Point at the settings chips and the inference time. Download the result.
2. **Hard-Routed**: blur high (probabilities, expert chosen), then "None" to show the identity bypass; upload an already-corrupted image.
3. **Soft MoE**: occlusion; show the stacked bar; then clean; run the temperature script in a terminal to show `tau`.
4. **Face-to-Sketch**: a face photo with Style 1, 2, 3 (webcam if you wish).
5. Show W&B (trial runs, a final run's charts and sample images), and the heatmap figure for Task 3.

### 14.3 If something goes wrong

| Problem | Cause and fix |
|---|---|
| `docker compose up` says it cannot connect to Docker | Docker Desktop is not running: start it, wait for the whale icon to settle |
| Port 8080 is already in use | Another copy is running: `docker compose down` in the other folder, or close the program using it |
| Status card shows fewer than 7 models loaded | Files missing in `models/`: run `python scripts/download_models.py`, then `docker compose restart backend` |
| A workspace says "Model ... is not loaded" | The same: a specific file is missing; the message names it |
| "Backend offline" banner | The backend container stopped: `docker compose logs backend`, then `docker compose up -d` |
| Webcam does not start | The browser needs camera permission for localhost; allow it, or use the upload instead |
| Upload rejected | Only PNG/JPG up to 10 MB are accepted (this is by design and shows the validation) |
| First request is slower | All models are loaded when the backend starts (the status card shows how many); the very first inference can still be a little slower while ONNX Runtime warms up, later requests are fast |
| Page looks old after a change | Rebuild: `docker compose up --build -d`, then hard-refresh the browser (`Ctrl+F5`) |

---

## Part 15: Where everything is

```
configs/            settings per task; *_best.yaml = what Optuna chose
src/data/           corruptions.py (the damage), prepare_pet.py, prepare_fs2k.py, manifests.py,
                    pet_dataset.py, fs2k_dataset.py, balanced.py
src/models/         autoencoder.py (Task 1 and specialists), classifier.py, hard_router.py,
                    soft_moe.py (Task 3), cgan.py (Task 4)
src/training/       losses.py, train_udae.py, train_classifier.py, train_specialists.py,
                    train_moe.py, train_cgan.py   (each: --mode optuna | final)
src/evaluation/     metrics.py, eval_udae.py, eval_task2.py, eval_moe.py, eval_cgan.py,
                    classification.py, routing_analysis.py, figures.py, plot_optuna.py, plot_curves.py
src/export/         export_onnx.py (--task task1..task4; verifies against PyTorch)
src/utils/          seed.py, config.py, tracking.py (W&B and Optuna helpers)
manifests/          pet_split.json, pet_val_manifest.jsonl, pet_test_manifest.jsonl, fs2k_split.json
optuna_studies/     one .db per study (5)
models/             the 7 ONNX files (not in git; GitHub Release)
checkpoints/        PyTorch weights (not in git)
notebooks/          Kaggle launchers
backend/            FastAPI app, tests, Dockerfile, sample images
frontend/           React + Tailwind app, Dockerfile, nginx.conf
docker-compose.yml  starts everything
scripts/            download_models.py, show_results.py, demo_temperature.py, build_gallery.py
report/             main.tex (IEEE paper), references.bib, results/ (CSV/JSON), figures/ (PNG; Stitch designs in figures/stitch/)
docs/               STUDY_GUIDE.md (this), testing_and_outputs_guide.md, decision_log.md,
                    ai_use_log.md, stitch_prompts.md, demo_video_script.md
tests/              automatic tests of the training code
```

To look at all figures with explanations, run `python scripts/build_gallery.py` and open `report/outputs_gallery.html` in a browser.

---

## Part 16: Your homework before the evaluation

The brief says you may be asked to explain or modify **any** submitted component. Reading the code yourself matters more than anything in this guide. Order of priority, with roughly how long:

| File | What to look for | Time |
|---|---|---|
| `src/data/corruptions.py` | the three apply functions; how the occlusion rectangles avoid overlapping; `corrupt_random` | 15 min |
| `src/models/autoencoder.py` | the shape comments; where the bottleneck is; what `skip=True` changes | 10 min |
| `src/training/losses.py` and `train_udae.py` (`train_one_epoch`, `fit`, `suggest_params`) | the training loop in about 20 lines; how a trial is scored and pruned | 20 min |
| `src/models/hard_router.py` | how clean images bypass the experts | 10 min |
| `src/models/soft_moe.py` | `forward`, the freezing logic, `moe_loss` (read the comments; they explain every step) | 20 min |
| `src/models/cgan.py` | where the style embedding enters the generator and the discriminator | 20 min |
| `src/training/train_cgan.py` (`train_one_epoch`) | the discriminator step and the generator step | 10 min |
| `backend/app/main.py` and `inference.py` | one endpoint from request to response | 20 min |
| `docker-compose.yml` and `frontend/nginx.conf` | what each service does and how they connect | 10 min |
| `docs/decision_log.md` | skim every entry; this is your evidence for "why" | 20 min |

Also do this:

1. **Run the app yourself with your own photos** (`docs/testing_and_outputs_guide.md`), including bad uploads.
2. **Open every figure** (the gallery script) and be able to say in one sentence what it shows and what it proves.
3. **Look at the AI-use log** (`docs/ai_use_log.md`). Several rows say "Pending: my own read-through". Do the reading, then tell Claude Code to update those rows, or reword them honestly as not done. Do not let a row claim a check you did not do.
4. **Confirm the facts that only you know**: the Kaggle GPU model (the report still has a TODO for it), the real deadline, and your name, roll number and email in the report and README.
5. **Rehearse the live demo once** from a fresh clone (Part 14).
