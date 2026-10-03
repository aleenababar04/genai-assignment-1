"""Build report/outputs_gallery.html: every figure with a plain-English explanation.

Run from the repository root:

    python scripts/build_gallery.py

then open report/outputs_gallery.html in a web browser. Figures that do not
exist (for example the app screenshots, before you take them) are skipped.
"""

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"
OUT = ROOT / "report" / "outputs_gallery.html"

# (file under report/figures, title, what it shows, how to read it, what it proves / what to say)
SECTIONS = [
    ("The data", [
        ("corruption_preview.png", "The corruptions used for testing",
         "Four test images (rows), each shown clean and then with salt-and-pepper noise, blur and occlusion at low, medium and high severity.",
         "Read each row left to right: the damage gets stronger within each group of three.",
         "These are the exact fixed corruptions of the test manifest, so every model is judged on the same damage."),
    ]),
    ("Task 1: universal denoising autoencoder", [
        ("task1_udae_examples.png", "Twelve representative test examples",
         "Columns: clean target, corrupted input, restored output, absolute error map. One typical image per condition and severity.",
         "On the error map, black means no error and bright yellow means a large error (fixed scale 0 to 0.5).",
         "Noise and blur are repaired with diffuse small errors on edges and fur. Occlusion boxes are filled with plausible colour blends, so the error concentrates in the box."),
        ("task1_udae_failures.png", "Four failure cases",
         "The lowest-SSIM test image of each condition.",
         "Same columns as above.",
         "The cat on a black background is 'repaired' even when clean, because black areas look like occlusion masks to a network that is not told the corruption. The flower-field dog shows dense texture that cannot pass the bottleneck. This failure motivates Task 2."),
        ("task1_udae_psnr_bars.png", "Quality before and after (PSNR)",
         "For each condition and severity: PSNR of the damaged input (blue) and of the restored output (orange).",
         "Orange above blue means the model helped. Higher is better.",
         "Strong damage improves a lot (noise, high occlusion). Clean and mild blur get worse: the output is capped near 25 dB by the bottleneck."),
        ("task1_udae_ssim_bars.png", "Quality before and after (SSIM)",
         "Same as the previous bars, using SSIM (0 to 1).",
         "Higher is better.",
         "Same story as PSNR; for low occlusion the SSIM is slightly lower than the input because the model softens the whole image."),
        ("task1_udae_curves.png", "Training curves of the final run",
         "Training loss, validation objective, validation PSNR and validation SSIM per epoch (80 epochs).",
         "The first two should fall, the last two should rise.",
         "Steady convergence, no overfitting (validation keeps improving with training), smoother as the learning rate decays."),
        ("task1_udae_optuna.png", "Optuna search for Task 1",
         "Left: the score of every trial (crosses are trials stopped early) and the best so far. Right: which hyperparameters mattered.",
         "Lower is better on the left. Longer bars on the right mean more influence.",
         "30 trials, 19 completed, 11 pruned. The learning rate mattered most (0.67), then dropout and alpha."),
        ("task1_udae_skip_examples.png", "Skip-connection ablation: examples",
         "The same twelve examples for the model with one 16x16 skip connection.",
         "Compare with the no-skip examples: look at fur and edges.",
         "Sharper outputs. The skip lets detail bypass the bottleneck, which is why we investigated it, and why we keep the skip-free model for the 'genuine bottleneck' requirement."),
        ("task1_udae_skip_failures.png", "Skip ablation: failure cases", "Worst cases of the skip model.", "As above.",
         "The black-background problem persists, so it is not only a bottleneck effect."),
        ("task1_udae_skip_psnr_bars.png", "Skip ablation: PSNR bars", "Before/after PSNR for the skip model.", "As above.",
         "Higher orange bars than the no-skip model for every condition."),
        ("task1_udae_skip_ssim_bars.png", "Skip ablation: SSIM bars", "Before/after SSIM for the skip model.", "As above.",
         "Higher than without the skip."),
        ("task1_udae_skip_curves.png", "Skip ablation: training curves", "Curves of the skip model's final run.", "As above.",
         "Converges like the main model, to a better level."),
    ]),
    ("Task 2: classifier, specialists, hard routing", [
        ("task2_test_confusion.png", "Classifier confusion matrix (test set)",
         "Rows are the true condition, columns the classifier's answer; each row sums to 1.",
         "A perfect classifier is a bright diagonal with zeros elsewhere.",
         "Almost perfect. The small off-diagonal numbers are low occlusion called clean and clean called blur or occlusion."),
        ("task2_classifier_val_confusion.png", "Classifier confusion matrix (validation set)",
         "Same matrix on the validation set, for the epoch that was kept.", "As above.",
         "Used for choosing the checkpoint; it agrees with the test result, so the model generalises."),
        ("task2_classifier_curves.png", "Classifier training curves",
         "Cross-entropy, accuracy (training and validation) and validation macro-F1 per epoch.",
         "Loss falls; accuracy and F1 rise towards 1.",
         "Converges within about 10 epochs; training and validation stay close, so no overfitting. One early dip (epoch 6) while the learning rate was high."),
        ("task2_classifier_optuna.png", "Optuna search for the classifier",
         "Trial scores (validation macro-F1) and hyperparameter importances.", "Higher is better here.",
         "12 trials, 3 completed, 9 pruned. Dropout, weight decay and learning rate matter; the network size hardly does (the small preset won)."),
        ("task2_specialists_optuna.png", "Optuna search for the specialists",
         "Trial scores (mean objective of the three specialists) and importances.", "Lower is better.",
         "10 trials, 3 completed, 7 pruned. Learning rate and the L1/SSIM weight dominate."),
        ("task2_specialist_salt_pepper_curves.png", "Salt-and-pepper specialist curves", "Training curves of this specialist.", "Loss falls, quality rises.", "Trained only on noisy images."),
        ("task2_specialist_blur_curves.png", "Blur specialist curves", "Training curves of this specialist.", "As above.", "Trained only on blurred images."),
        ("task2_specialist_occlusion_curves.png", "Occlusion specialist curves", "Training curves of this specialist.", "As above.", "Trained only on images with black boxes."),
        ("task2_examples.png", "Twelve examples with predicted routing",
         "Clean target, corrupted input, restored output and error map; the row label names the expert chosen.",
         "Check that each corruption goes to its own expert and clean images are untouched.",
         "Hard routing works: the right expert is picked and clean images pass through unchanged."),
        ("task2_misrouted.png", "The six worst misrouted images",
         "Images sent to the wrong branch, with the biggest quality loss compared with oracle routing.",
         "Row label: true class, branch chosen, PSNR (and the PSNR it would have had).",
         "All are clean photos with large black areas (black background, black borders). The classifier reads them as occlusion and the occlusion expert paints over them."),
        ("task2_predicted_psnr_bars.png", "Predicted-routing quality before and after",
         "PSNR of input and output for every condition and severity under the real (predicted) routing.", "Orange above blue means it helped.",
         "Almost identical to oracle routing; clean images are intact."),
    ]),
    ("Task 3: soft mixture-of-experts", [
        ("task3_moe_routing_heatmap.png", "Routing heatmap (the key figure of Task 3)",
         "Average weight given to each of the four branches (columns) for each true corruption and severity (rows).",
         "Brighter means more weight. A bright diagonal means the gate sends each corruption to its own expert.",
         "Each corruption goes to its own expert and the routing sharpens as severity grows. For mild damage the leftover weight goes to the identity branch, never to a wrong expert."),
        ("task3_moe_weight_distribution.png", "Weight distribution per corruption",
         "Box plots of the four branch weights for each true corruption.",
         "The box is the middle half of the images; the line inside is the median; dots are outliers.",
         "Shows the spread across individual images, not just the average: sharing is image-by-image, not a constant split."),
        ("task3_moe_dominant_examples.png", "Examples where one expert dominates",
         "Four images where the gate is very sure; row labels give the four weights (id, sp, bl, oc).", "A weight near 1.00 means that branch alone does the work.",
         "Strong corruptions get a single expert."),
        ("task3_moe_distributed_examples.png", "Examples where the weight is shared",
         "Four images with the most evenly spread weights.", "Look at the weights in the row labels.",
         "Mild or ambiguous cases (a low occlusion on a black cat, light noise on grass) share weight, and mixing keeps their detail."),
        ("task3_moe_examples.png", "Twelve representative examples", "Clean, corrupted, restored, error map with the weights.", "As in Task 1.",
         "The soft mixture's typical behaviour across all conditions."),
        ("task3_moe_failures.png", "Failure cases with routing weights",
         "The worst image of each condition, with the weights in the row labels.", "Compare the weights with the true corruption.",
         "Separates routing errors (clean cat sent 0.87 to the occlusion expert) from expert errors (correct routing, but the expert cannot recover the content)."),
        ("task3_moe_psnr_bars.png", "Quality before and after", "PSNR of input and output per condition and severity.", "Orange above blue means it helped.",
         "Best of the three systems on mild corruptions, and clean images stay almost intact."),
        ("task3_moe_curves.png", "Training curves",
         "Loss parts, validation objective, gate accuracy and the mean weight of each branch per epoch (epochs 1-2 warm-up, then joint fine-tuning).",
         "Mean branch weights should stay away from 0 (no collapse).",
         "The objective falls through fine-tuning; no branch collapses; the identity weight rises as the gate learns to hedge."),
        ("task3_moe_optuna.png", "Optuna search for Task 3", "Trial scores and importances.", "Lower is better.",
         "8 trials, all completed, none pruned for collapse; importances are spread evenly."),
    ]),
    ("Task 4: face-to-sketch conditional GAN", [
        ("task4_cgan_style_swap.png", "Style swap (proof that the style works)",
         "Five test photos, each drawn in Style 1, Style 2 and Style 3.",
         "Compare columns for the same row.",
         "Same identity, pose and hair, but a visibly different drawing style: light outline, heavy shading, soft tone. The learned style embedding controls the output."),
        ("task4_cgan_examples.png", "Test examples", "Photo, true sketch and generated sketch, three examples per style.", "Compare the last two columns.",
         "The generator reproduces structure and style; stroke details differ from the artist's."),
        ("task4_cgan_failures.png", "Four failure cases", "The four test pairs with the lowest SSIM (all Style 2).", "Compare true and generated sketches.",
         "Recognisable and in the right style, but the artist's individual hair strokes sit elsewhere. These failures are partly a limit of pixel-wise metrics."),
        ("task4_cgan_curves.png", "Training curves",
         "Discriminator real and fake loss, generator adversarial loss, generator L1, validation SSIM and objective per epoch.",
         "Healthy GAN training: the losses oscillate but do not blow up.",
         "The discriminator slowly gains the upper hand (normal), nothing collapsed, and validation SSIM was still creeping up at epoch 100."),
        ("task4_cgan_optuna.png", "Optuna search for Task 4", "Trial scores and importances.", "Lower is better.",
         "8 trials, 3 completed, 5 pruned; the balance between the two learning rates and dropout matter most."),
    ]),
    ("Google Stitch design", [
        ("stitch/00_design_system.png", "Design system", "The style guide generated in Stitch.", "Colours, type, components.", "Evidence that the design started in Stitch."),
        ("stitch/01_overview.png", "Overview screen", "The landing page with four workspace cards.", "", ""),
        ("stitch/02_universal_restoration.png", "Universal Restoration screen", "Input panel and result panel.", "", ""),
        ("stitch/03_hard_routed_restoration.png", "Hard-Routed Restoration screen", "Classifier probabilities and chosen expert.", "", ""),
        ("stitch/04_soft_moe_restoration.png", "Soft Mixture-of-Experts screen", "Routing weights and contribution bar.", "", ""),
        ("stitch/05_face_to_sketch.png", "Face-to-Sketch screen", "Upload, style choice, side-by-side result.", "", ""),
        ("stitch/06_face_to_sketch_webcam.png", "Webcam state", "Live preview and capture.", "", ""),
        ("stitch/07_loading_state.png", "Loading state", "Spinner and disabled controls while the model runs.", "", ""),
        ("stitch/08_error_state.png", "Error and empty state", "Unsupported file message and empty results.", "", ""),
    ]),
    ("The running application (your screenshots)", [
        ("app/01_universal.png", "Universal Restoration in the app", "", "", ""),
        ("app/02_hard_routed.png", "Hard-Routed Restoration in the app", "", "", ""),
        ("app/03_soft_moe.png", "Soft Mixture-of-Experts Restoration in the app", "", "", ""),
        ("app/04_face_to_sketch.png", "Face-to-Sketch Generator in the app", "", "", ""),
    ]),
]

STYLE = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:16px;background:#f8fafc;color:#0f172a}
h1{font-size:26px} h2{margin-top:40px;border-bottom:2px solid #4f46e5;padding-bottom:4px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:18px 0}
.card h3{margin:0 0 8px 0;font-size:17px} .card img{max-width:100%;border:1px solid #e2e8f0;border-radius:6px}
.label{font-weight:600;color:#4f46e5} .file{font-family:monospace;font-size:12px;color:#64748b}
p{margin:6px 0} nav a{margin-right:12px}
@media (prefers-color-scheme:dark){body{background:#0f172a;color:#f1f5f9}.card{background:#1e293b;border-color:#334155}
.card img{border-color:#334155}.file{color:#94a3b8}.label{color:#a5b4fc}}
"""


def card(rel_path, title, shows, read, meaning):
    parts = [f'<div class="card"><h3>{html.escape(title)}</h3>',
             f'<img src="figures/{html.escape(rel_path)}" alt="{html.escape(title)}">',
             f'<p class="file">report/figures/{html.escape(rel_path)}</p>']
    for label, text in (("What it shows", shows), ("How to read it", read), ("What it means", meaning)):
        if text:
            parts.append(f'<p><span class="label">{label}:</span> {html.escape(text)}</p>')
    parts.append("</div>")
    return "\n".join(parts)


def main():
    body, nav, shown = [], [], 0
    for index, (section, items) in enumerate(SECTIONS):
        cards = [card(*item) for item in items if (FIGURES / item[0]).exists()]
        if not cards:
            continue
        shown += len(cards)
        nav.append(f'<a href="#s{index}">{html.escape(section)}</a>')
        body.append(f'<h2 id="s{index}">{html.escape(section)}</h2>\n' + "\n".join(cards))
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Project outputs gallery</title><style>{STYLE}</style></head><body>"
            f"<h1>Project outputs gallery</h1><p>Every figure produced by the project, with an explanation. "
            f"Numbers behind them: <code>python scripts/show_results.py</code>.</p><nav>{' '.join(nav)}</nav>"
            + "\n".join(body) + "</body></html>")
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} with {shown} figures")


if __name__ == "__main__":
    main()
