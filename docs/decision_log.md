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

### Task 1: architecture and bottleneck

- **Date:** 2026-10-03
- **Decision:** A convolutional autoencoder with four stride-2 stages (128 -> 64 -> 32 -> 16 -> 8, channels c, 2c, 4c, 8c), a spatial bottleneck of 8 x 8 x `latent_channels`, and a decoder of four "nearest-neighbour upsample + convolution" stages ending in a sigmoid. No skip connections in the main model. `latent_channels` is the bottleneck dimension tuned by Optuna.
- **Question:** What form should the "genuine compressed latent representation" take, and how should the decoder upsample?
- **Alternatives considered:**
  - Flatten to one latent vector through a linear layer: the strongest compression, but it discards spatial layout and adds a large linear layer.
  - Spatial bottleneck (8 x 8 x C): still compressed (48x, 24x, 12x and 6x fewer values than the 49,152 input values for C = 16, 32, 64, 128) and keeps layout.
  - Transposed convolutions in the decoder: learnable upsampling, but prone to checkerboard artefacts.
  - Upsample then convolve: avoids those artefacts and exports cleanly to ONNX.
  - Full U-Net skips: best pixel accuracy, but the brief rules out unrestricted skip connections because the input can then be copied past the bottleneck.
- **Sources consulted:** Assignment brief, Task 1 (bottleneck and skip-connection requirements). To read first-hand and cite before the report: Odena et al., "Deconvolution and Checkerboard Artifacts" (Distill, 2016), for the upsample-plus-convolution choice; Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" (ICML 2008), for the denoising-autoencoder idea.
- **Experiment and numbers:** Parameter counts (no skip): 1,865,891 for (c = 32, latent 32); 4,132,835 for (48, 32); 7,289,123 for (64, 32); 7,780,739 for (64, 128). `tests/test_autoencoder.py`: 17 passed, including "decode(encode(x)) equals forward(x)", which shows the output depends on the input only through the bottleneck. Trained results: TODO after the Kaggle run.
- **Choice and why:** Spatial bottleneck with upsample-plus-convolution and no skips. It satisfies the bottleneck rule without any argument, and the compression ratio is an explicit, tunable number.
- **Report section it feeds:** Task 1 methodology, architecture.

### Task 1: limited skip connection as an ablation only

- **Date:** 2026-10-03
- **Decision:** The model has an optional single skip connection at the 16 x 16 level (`skip=True`), off by default. It is trained once with the best configuration as an ablation and compared with the main model.
- **Question:** Should limited skip connections be used, given that the brief allows them only if their purpose and effect are investigated?
- **Alternatives considered:**
  - No skip at all: nothing to justify, but no evidence about what a skip would change.
  - One skip at 16 x 16 (4c channels): low resolution, so it cannot copy fine detail, but it does let information bypass the 8 x 8 bottleneck.
  - Skips at high resolution (64 x 64 or 128 x 128): would let the network copy the input, which the brief forbids.
- **Sources consulted:** Assignment brief, Task 1: "If limited skip connections are used, their purpose and effect must be investigated and justified in the report."
- **Experiment and numbers:** TODO after the Kaggle run: test PSNR/SSIM per condition for `task1_udae` against `task1_udae_skip`. The skip model has 1,939,619 parameters at (c = 32, latent 32).
- **Choice and why:** Main model without skips; the skip variant is reported as an ablation so its effect is measured, not assumed.
- **Report section it feeds:** Task 1 results, ablation.

### Task 1: Optuna search design and validation objective

- **Date:** 2026-10-03
- **Decision:** TPE sampler (seed 42) with a median pruner (5 start-up trials, 3 warm-up epochs), 30 trials of 10 epochs. Search space: learning rate 1e-4 to 3e-3 (log), batch size {32, 64, 128}, bottleneck channels {16, 32, 64, 128}, base channels {32, 48, 64}, dropout 0 to 0.3, alpha 0.5 to 0.95. Every trial is scored on the validation manifest with the fixed objective `0.8 * L1 + 0.2 * (1 - SSIM)`.
- **Question:** How can trials that train with different loss weights be compared fairly?
- **Alternatives considered:**
  - Use each trial's own training loss on the validation set: not comparable, because alpha changes the scale of the loss, so Optuna would simply favour whichever alpha makes the number small.
  - A fixed objective with the brief's starting weights (0.8 / 0.2): the same yardstick for every trial; combines reconstruction quality and structural similarity as the brief requires. A trial with alpha near 0.8 trains on almost what is measured, which may favour it slightly.
  - Multi-objective search (L1 and SSIM separately): no weighting needed, but returns a Pareto front instead of one best trial, which the brief asks for.
- **Sources consulted:** Assignment brief, Task 1: the study must cover learning rate, batch size, bottleneck dimension, encoder channels, dropout and alpha, and "the validation objective should combine reconstruction quality and structural similarity". To read first-hand and cite: Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (KDD 2019), and the Optuna documentation on `TPESampler` and `MedianPruner`.
- **Experiment and numbers:** TODO after the Kaggle run: number of completed and pruned trials, best trial, best configuration. PSNR, SSIM and L1 are stored with every trial as user attributes.
- **Choice and why:** Fixed 0.8 / 0.2 objective with TPE and median pruning. One comparable number per trial, with the plain metrics kept alongside so the choice can be re-examined.
- **Report section it feeds:** Task 1 methodology, Optuna search design.

### Task 2: classifier architecture

- **Date:** 2026-10-03
- **Decision:** Four convolutional blocks (two 3x3 conv + BatchNorm + ReLU, then 2x2 max-pooling), global average pooling, dropout and a linear layer to 4 logits. The first block runs at the full 128 x 128 resolution. Channel width is one of three presets tuned by Optuna: small (16-128, 294,516 parameters), medium (32-256, 1,174,244), large (48-384, 2,639,188).
- **Question:** What classifier can separate clean, salt-and-pepper, blur and occlusion inputs, and also serve as the Task 3 gate?
- **Alternatives considered:**
  - Downsample early (stride-2 first layer, or a pretrained ImageNet network on resized input): cheaper, but the evidence that separates clean from mildly blurred or lightly noisy images is in single pixels and fine edges, which early downsampling removes.
  - Full-resolution first block, then pooling: keeps that detail; costs more computation in the first block.
  - Output probabilities directly (softmax inside the model): convenient for display, but cross-entropy and the Task 3 temperature softmax both need raw logits.
- **Sources consulted:** Assignment brief, Task 2 (four classes, cross-entropy, Optuna over learning rate, batch size, channel configuration, dropout and weight decay) and Task 3 (gate initialised from the classifier, `softmax(G(x)/tau)`).
- **Experiment and numbers:** `tests/test_classifier_balanced.py`, 17 passed, including an ONNX export that matches PyTorch within 1e-4. CPU forward for a batch of 64 (medium): about 1.6 s, so training must run on the GPU. Trained results: TODO after the Kaggle run (expect clean vs. low blur to be the hardest pair).
- **Choice and why:** Full-resolution first block with logits output, for the fine-detail reason above and so the same network can become the gate.
- **Report section it feeds:** Task 2 methodology, classifier.

### Task 2: exactly balanced training batches

- **Date:** 2026-10-03
- **Decision:** A custom collate function assigns labels 0,1,2,3,0,1,2,3,... across each batch, shuffles them, and corrupts every image according to its label. Batch sizes must be multiples of 4, and the last incomplete batch is dropped, so every batch has exactly 25% of each class. The same batching will be reused for Task 3.
- **Question:** How should "the training batches must be balanced" be met?
- **Alternatives considered:**
  - Random condition per image with probability 1/4 (the Task 1 dataset): balanced only on average; a batch of 32 can easily hold 4 of one class and 12 of another.
  - A weighted sampler: also balanced only in expectation.
  - Assign labels by batch position: exact balance in every batch, and the shuffle keeps position from carrying information.
- **Sources consulted:** Assignment brief, Task 2, and Task 3's balance loss, which is defined over "a balanced training batch".
- **Experiment and numbers:** Tests confirm exact counts of B/4 per class for batch sizes 4, 8 and 64 and over a full epoch. Collate cost on CPU: about 0.10 s per batch of 64.
- **Choice and why:** Label-by-position collate, the only option that guarantees balance per batch rather than on average.
- **Report section it feeds:** Task 2 methodology, training procedure.

### Task 2: classifier Optuna objective

- **Date:** 2026-10-03
- **Decision:** Maximise validation macro-F1 on the balanced validation manifest; checkpoint selection uses macro-F1 with ties broken by the lower validation cross-entropy. Search space: learning rate 1e-4 to 3e-3 (log), batch size {32, 64, 128}, channel preset {small, medium, large}, dropout 0 to 0.5, weight decay 1e-6 to 1e-3 (log, AdamW). 25 trials of 8 epochs with median pruning.
- **Question:** Which validation number should the classifier search optimise?
- **Alternatives considered:**
  - Accuracy: on a perfectly balanced validation set it ranks models almost identically to macro-F1, but macro-F1 is one of the required reported metrics and penalises a model that sacrifices one class.
  - Validation cross-entropy: smoother, but rewards confident probabilities rather than correct decisions; kept as the tie-breaker and logged.
- **Sources consulted:** Assignment brief, Task 2 (required metrics). To read first-hand: Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019), for AdamW.
- **Experiment and numbers:** TODO after the Kaggle run.
- **Choice and why:** Macro-F1, because it is what the report must show and it cannot hide a weak class.
- **Report section it feeds:** Task 2 methodology, Optuna search design.

### Task 2: shared search for the three specialists

- **Date:** 2026-10-03
- **Decision:** One Optuna study for all three specialists. In each trial the three are trained side by side (one epoch each in turn), each only on its own corruption, and the trial's score is the mean of their validation objectives (fixed `0.8 * L1 + 0.2 * (1 - SSIM)`, as in Task 1, each on its own 736 validation items). The best shared configuration is then used to train the three independently, each with its own W&B run and checkpoint. Search space: learning rate, batch size, bottleneck channels, base channels and alpha (dropout fixed at 0).
- **Question:** How can the specialists be tuned without running three full searches?
- **Alternatives considered:**
  - Three separate searches: the most tailored settings, at three times the cost.
  - One search on a single proxy model trained on all three corruptions together: cheap, but tunes a different model from the ones actually used.
  - One search scoring all three real specialists: the brief explicitly allows a shared search; costs three short trainings per trial, and side-by-side training still allows per-epoch pruning on the mean.
- **Sources consulted:** Assignment brief, Task 2: "you may use a shared Optuna search to identify a common architecture and then train the three specialists independently".
- **Experiment and numbers:** TODO after the Kaggle run (the best trial's per-specialist PSNR/SSIM are stored as trial attributes).
- **Choice and why:** Shared search over the real specialists, which is what the brief allows and tunes exactly the models that get used.
- **Report section it feeds:** Task 2 methodology, specialists.

### Task 2: routing implementation and evaluation

- **Date:** 2026-10-03
- **Decision:** `HardRoutedRestorer` takes the argmax of the classifier's softmax, returns clean-routed images unchanged (identity bypass), and runs each expert only on the images routed to it. It is evaluated on the test manifest in oracle mode (manifest label chooses the branch) and predicted mode. Misrouted images are tabulated by (true class, chosen branch) with their PSNR loss against oracle routing, and the worst six are shown. For the application, the classifier and the three specialists are exported as four ONNX files and the backend performs the routing.
- **Question:** How should routing be implemented, evaluated and deployed?
- **Alternatives considered:**
  - Run all three experts on every image and pick one output: simpler code, but three times the work and the clean image would still pass through an expert, against the identity-bypass rule.
  - Run only the chosen expert per image: no wasted computation, and clean images are bit-identical to the input.
  - Export the whole router as one ONNX graph: possible, but data-dependent branching is awkward in ONNX, and the brief asks for the classifier and specialists to be exported individually.
- **Sources consulted:** Assignment brief, Task 2 (routing equation, identity bypass, oracle vs. predicted modes, export of the four models).
- **Experiment and numbers:** `tests/test_hard_router.py`, 14 passed: clean-routed images are returned unchanged, an expert is never called when nothing is routed to it, and oracle routes override the classifier while probabilities are still reported. Test-set results: TODO after the Kaggle run.
- **Choice and why:** Masked per-expert routing with four separate ONNX files; it follows the brief's equation literally and keeps the backend simple.
- **Report section it feeds:** Task 2 methodology and results.

### Task 3: mixture design, initialisation and two-stage training

- **Date:** 2026-10-03
- **Decision:** `SoftMoERestorer` computes `w = softmax(G(x) / tau)` with G the Task 2 classifier network, runs all three experts on every image, and outputs `w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occ(x)`. Gate and experts are loaded from the trained Task 2 checkpoints. Stage 1 (warm-up): experts frozen and kept in eval mode, so their BatchNorm statistics do not drift; only the gate trains (Adam, lr 5e-4). Stage 2: everything is unfrozen and fine-tuned with the Optuna-chosen smaller learning rate and a cosine schedule. Batches are exactly balanced (same collate as Task 2). `tau` is stored in the model, so it is saved with the weights and exported inside the ONNX graph.
- **Question:** How should the hard-routing system be turned into a differentiable mixture that trains stably?
- **Alternatives considered:**
  - Train gate and experts from scratch together: the brief forbids it, and a random gate would give the experts meaningless mixtures to learn from.
  - Freeze the experts by turning off gradients only: BatchNorm running statistics would still change in train mode, so the "frozen" experts would drift; keeping them in eval mode prevents this.
  - Run only the top-weighted experts (sparse mixture): cheaper, but no longer the fully weighted sum the brief defines.
- **Sources consulted:** Assignment brief, Task 3 (gating equation, initialisation, warm-up then joint fine-tuning, loss terms). To read first-hand and cite: Jacobs et al., "Adaptive Mixtures of Local Experts" (Neural Computation, 1991); Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017), on load balancing.
- **Experiment and numbers:** `tests/test_soft_moe.py`, 18 passed: mixing matches the hand-computed weighted sum, frozen experts get no gradients and keep their BatchNorm statistics, and the full pipeline exports to one ONNX graph matching PyTorch within 1e-4 on output, weights and logits. Trained results: TODO after the Kaggle run.
- **Choice and why:** As above. It follows the brief's equations exactly and avoids the BatchNorm drift that a gradient-only freeze would cause.
- **Report section it feeds:** Task 3 methodology.

### Task 3: loss terms, cross-entropy on tempered logits, and collapse pruning

- **Date:** 2026-10-03
- **Decision:** `L = l1*L1 + ls*(1 - SSIM) + lc*CE + lb*L_balance` with `ls = 1 - l1`, `L_balance = sum_k (wbar_k - 1/4)^2` over a balanced batch (the brief's suggested form), and the cross-entropy computed on `G(x) / tau`, the same tempered distribution that produces the routing weights. Optuna (20 trials, 2 warm-up + 6 fine-tune epochs) searches the fine-tuning learning rate (1e-5 to 3e-4, log), tau (0.5 to 3.0), lc (0.01 to 1, log), lb (0.001 to 0.1, log) and l1 (0.5 to 0.95). Trial 0 is the brief's starting point (0.8, 0.2, 0.1, 0.01, tau 1). A trial is pruned for routing collapse if, after the warm-up, any branch's mean weight on the balanced validation set falls below 0.05; otherwise the median pruner applies. Trials are scored with the same fixed objective as Tasks 1 and 2.
- **Question:** How should the joint loss and its search be set up so the gate stays meaningful and does not collapse onto one expert?
- **Alternatives considered:**
  - Cross-entropy on the raw logits: trains the classifier but not the routing weights actually used when tau is not 1.
  - Cross-entropy on `logits / tau`: directly ties the weights used for mixing to the known corruption label.
  - Entropy regulariser instead of the squared balance term: the brief allows it if justified; the squared term was kept because it is the brief's suggestion and is easy to interpret (zero when each branch averages 1/4 on a balanced batch).
  - Detect collapse only after training: wastes the trial's compute; pruning stops it as soon as it shows.
- **Sources consulted:** Assignment brief, Task 3 (loss, starting weights, "trial pruning may be used when a configuration performs poorly or exhibits routing collapse").
- **Experiment and numbers:** TODO after the Kaggle run: number of trials pruned for collapse, the best lambdas and tau, and how the mean weights per true corruption compare with the ideal (about 1 on the matching branch).
- **Choice and why:** As above. The brief's loss with the CE term on the tempered distribution, plus explicit collapse pruning.
- **Report section it feeds:** Task 3 methodology and Optuna search design.

### Task 3: gate analysis

- **Date:** 2026-10-03
- **Decision:** On the test manifest, report the mean of each branch weight per (true corruption, severity) as a CSV and a heatmap, the spread of the weights per true corruption as box plots, four "one expert dominates" examples (highest max weight, spread across corruptions) and four "weight shared" examples (highest routing entropy). An expert counts as **inactive** if it is the top-weighted branch for under 1% of images and its mean weight is below 0.05, and as **dominating unrelated inputs** if its mean weight on images of other corruptions exceeds 0.5.
- **Question:** How can the brief's gate-behaviour requirements be checked objectively rather than by eye?
- **Alternatives considered:** Inspecting a few images only (subjective); thresholds on fixed numbers (reproducible and reportable). The thresholds are a judgement call and are stated in the report so they can be challenged.
- **Sources consulted:** Assignment brief, Task 3 (average weights per corruption and severity, dominant vs. distributed examples, heatmap, inactive or dominating experts).
- **Experiment and numbers:** `tests/test_routing_analysis.py`, 10 passed, including hand-made healthy, inactive and dominating cases. Results: TODO after the Kaggle run.
- **Choice and why:** Numeric criteria plus figures, so every claim about the gate is backed by a number.
- **Report section it feeds:** Task 3 results, routing analysis.

### Task 4: FS2K split, pairing and paired augmentation

- **Date:** 2026-10-03
- **Decision:** Use the official `anno_train.json` / `anno_test.json` (1,058 / 1,046 pairs according to the FS2K repository). Hold out 15% of the training records per style (seed 42, `round(n_style * 0.15)`) as validation and save the split to `manifests/fs2k_split.json`. A photo path maps to its sketch by the official rule (photo/photoK/imageN -> sketch/sketchK/sketchN, `.jpg` first, then `.png`). Both images go through the same RGB conversion and direct bicubic resize to 128 x 128. Training augmentation is pix2pix-style jitter (resize to 143 x 143, random 128 x 128 crop) plus a horizontal flip, applied to the stacked photo-sketch pair so both always receive identical parameters.
- **Question:** How should the data be split and augmented without breaking the photo-sketch correspondence?
- **Alternatives considered:**
  - Random (unstratified) validation split: could leave a style under-represented in validation, so per-style quality would be measured on very few images.
  - Augment photo and sketch independently: destroys pixel alignment, which the brief explicitly forbids.
  - No augmentation: simpler, but about 900 training pairs is little for a GAN; jitter and flip are the pix2pix defaults.
- **Sources consulted:** Assignment brief, Task 4 (official split, 15% stratified validation with seed 42, paired augmentation). The FS2K GitHub repository (github.com/DengPingFan/FS2K), README and `tools/split_train_test.py`, read via the tool on 2026-10-03 for the folder layout, annotation fields and the photo-to-sketch naming rule. To cite: Fan et al., "FS2K" (the dataset paper named in the repository).
- **Experiment and numbers:** `tests/test_fs2k.py`, 17 passed on a synthetic FS2K tree, including the check that an augmented pair whose sketch is an exact copy of the photo stays identical over 50 draws. Real per-style counts: TODO after running `python -m src.data.prepare_fs2k` (it also prints the original image sizes and how many sketches are greyscale, which decides whether the generator should output 1 or 3 channels; currently 3).
- **Choice and why:** As above, to follow the brief exactly and keep the pairing provably intact.
- **Report section it feeds:** Task 4 dataset preparation.

### Task 4: generator, discriminator and style conditioning

- **Date:** 2026-10-03
- **Decision:** pix2pix-style U-Net generator for 128 x 128 (7 stride-2 downsamplings to 1 x 1, skip connections, dropout in the three innermost up-layers, tanh output) and a 70 x 70 PatchGAN discriminator (C64-C128-C256-C512, output a 14 x 14 map of real/fake logits for a 128 x 128 input). Each network has its OWN learned `nn.Embedding(3, style_dim)`. The generator receives the style twice: as a broadcast map concatenated with the photo at the input, and concatenated with the 1 x 1 bottleneck. The discriminator receives it as a broadcast map concatenated with the photo and the (real or generated) sketch. BatchNorm everywhere except the first encoder layer and the 1 x 1 bottleneck (where it would be undefined at batch size 1). Weights initialised N(0, 0.02).
- **Question:** How should the style condition be built into both networks, as the brief requires, rather than used only as an interface label?
- **Alternatives considered:**
  - One-hot style channels instead of a learned embedding: the brief asks for a learned categorical embedding.
  - Style injected only at the input: early layers can use it for stroke texture, but the global style decision is then made only through the skip-free path; adding it at the bottleneck gives the decoder direct access.
  - Feature-wise modulation (FiLM/AdaIN) in every layer: more expressive, but more code to explain and not required.
  - Shared embedding between G and D: would couple the two players' parameters; separate embeddings keep the adversarial game clean.
  - InstanceNorm instead of BatchNorm: a known alternative for style tasks; BatchNorm was kept to match pix2pix.
- **Sources consulted:** Assignment brief, Task 4. To read first-hand and cite: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017); Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014).
- **Experiment and numbers:** `tests/test_cgan.py`, 16 passed: the style changes the generator output and the discriminator logits, gradients reach both embeddings, batch size 1 works, and the export wrapper's ONNX graph matches PyTorch within 1e-4 for all three styles. Parameters (style_dim 16): generator 10,535,571 / 41,977,075 and discriminator 704,465 / 2,785,137 at base channels 32 / 64.
- **Choice and why:** As above, the standard paired image-to-image design with the condition visibly built into both networks.
- **Report section it feeds:** Task 4 methodology.

### Task 4: losses, Optuna objective and training schedule

- **Date:** 2026-10-03
- **Decision:** Discriminator loss `0.5 * (BCE(D(x,y,s), 1) + BCE(D(x,G(x,s),s), 0))` with logits; generator loss `BCE(D(x,G(x,s),s), 1) + lambda_L1 * L1`. Adam with betas (0.5, 0.999). The four losses (D real, D fake, G adversarial, G L1) are logged separately every epoch, and the same validation photos (two per style) are logged every 10 epochs as photo / true sketch / generated / the same photo in all three styles. Optuna (12 trials of 25 epochs, trial 0 = pix2pix defaults) tunes both learning rates, batch size {1, 4, 8, 16}, base channels {32, 48, 64}, dropout 0-0.5, style embedding size {8, 16, 32} and lambda_L1 10-200 (log). The best configuration is retrained for 200 epochs. Each trial is scored with the fixed validation objective `0.5 * L1 + 0.5 * (1 - SSIM)` on [0, 1] images; the generator is evaluated with dropout off.
- **Question:** How can a GAN be tuned and selected when its losses do not measure output quality?
- **Alternatives considered:**
  - Select on the generator or discriminator loss: these oscillate by design and do not track image quality.
  - FID: the standard GAN metric, but with only about 160 validation images it is strongly biased and noisy, and it needs an Inception network.
  - L1 + SSIM against the paired ground truth: possible because FS2K is paired; rewards correct structure, though it can favour slightly smoother sketches. This limitation is stated in the report.
- **Sources consulted:** Assignment brief, Task 4 (BCE with logits, initial lambda_L1 = 100, the hyperparameters to tune, shorter Optuna trials then full retraining, separate loss logging, fixed validation samples). To read first-hand: the pix2pix paper (loss halving for D, Adam betas, lambda 100) and Heusel et al., "GANs Trained by a Two Time-Scale Update Rule" (NeurIPS 2017) for FID and its sample-size bias.
- **Experiment and numbers:** TODO after the Kaggle run.
- **Choice and why:** Paired L1/SSIM objective, because it is meaningful with a small validation set and is the same kind of measure used in Tasks 1-3.
- **Report section it feeds:** Task 4 methodology, Optuna search design and limitations.

## Upcoming decisions

Empty headings for decisions that still have to be made. Fill each one in with the template above when the decision is taken.

### Task 1: Universal denoising autoencoder

#### SSIM implementation and window size

Currently `pytorch-msssim` with its defaults (11 x 11 Gaussian window, sigma 1.5, data range 1.0), used for both the loss and the metric. TODO: record why this implementation and window were kept, after reading Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP, 2004).

#### Loss weight alpha between L1 and (1 - SSIM)

TODO: fill in from the Optuna result (best alpha and how the objective varies with it).

### Task 3: Soft mixture-of-experts

#### Gate temperature and loss weights

TODO: fill in from the Optuna result (best tau, lambdas, and how routing sharpness changed with tau).

### Task 4: Style-conditioned face-to-sketch GAN

#### Final lambda_L1 and output channels

TODO: fill in from the Optuna result and the FS2K greyscale check.

### Cross-cutting

#### ONNX opset version and verification tolerance

Opset 17 with the classic TorchScript exporter (`dynamo=False`, needs no extra packages; PyTorch warns it is deprecated), dynamic batch axis, and a maximum absolute PyTorch-ONNX difference of 1e-4 on real validation images (random photos for Task 4). TODO: report the measured differences for the trained models (written to `report/results/*_onnx_check.json`).

#### How trained models are distributed to the evaluator

TODO:

#### Frontend and backend structure (one app, four workspaces)

TODO:
