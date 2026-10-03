// Names, colours and labels shared by several components.
//
// Tailwind only generates classes it can find written out in full in the
// source code, so every colour class below is spelled out completely.

// The four categories. The classifier classes ("clean", ...) and the soft-MoE
// branches ("identity", ...) share colours: clean <-> identity.
// Colours follow the Stitch "model degradation categories":
// Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.
export const CATEGORIES = {
  clean: {
    label: "Clean",
    expert: "Identity / clean branch",
    bar: "bg-emerald-500",
    text: "text-emerald-700",
    accent: "text-emerald-500",
    soft: "bg-emerald-50",
    border: "border-emerald-500",
    softBorder: "border-emerald-200",
    track: "bg-emerald-100",
  },
  salt_pepper: {
    label: "Salt-and-pepper",
    expert: "Salt-and-pepper expert",
    bar: "bg-amber-500",
    text: "text-amber-700",
    accent: "text-amber-500",
    soft: "bg-amber-50",
    border: "border-amber-500",
    softBorder: "border-amber-200",
    track: "bg-amber-100",
  },
  blur: {
    label: "Blur",
    expert: "Blur expert",
    bar: "bg-cyan-500",
    text: "text-cyan-700",
    accent: "text-cyan-500",
    soft: "bg-cyan-50",
    border: "border-cyan-500",
    softBorder: "border-cyan-200",
    track: "bg-cyan-100",
  },
  occlusion: {
    label: "Occlusion",
    expert: "Occlusion expert",
    bar: "bg-fuchsia-500",
    text: "text-fuchsia-700",
    accent: "text-fuchsia-500",
    soft: "bg-fuchsia-50",
    border: "border-fuchsia-500",
    softBorder: "border-fuchsia-200",
    track: "bg-fuchsia-100",
  },
};
// The soft-MoE "identity" branch uses the clean colours.
CATEGORIES.identity = CATEGORIES.clean;

// Order used for bars and cards.
export const CLASS_ORDER = ["clean", "salt_pepper", "blur", "occlusion"];
export const BRANCH_ORDER = ["identity", "salt_pepper", "blur", "occlusion"];

// Corruption options of the input panel (values are the backend form values).
// `short` and `hint` are the two lines shown on the option cards; `category` picks the colour dot.
export const CORRUPTIONS = [
  { value: "none", label: "None", short: "None", hint: "Ground truth", category: "clean" },
  { value: "salt_pepper", label: "Salt-and-pepper noise", short: "Salt & Pepper", hint: "Impulsive", category: "salt_pepper" },
  { value: "blur", label: "Gaussian blur", short: "Gaussian Blur", hint: "Gaussian kernel", category: "blur" },
  { value: "occlusion", label: "Rectangular occlusion", short: "Occlusion", hint: "Synthetic boxes", category: "occlusion" },
];

export const SEVERITIES = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
];

// Navigation: hash route -> page title. The workspace names must stay exact.
// `icon` is a Material Symbols name (as in the Stitch screens).
export const PAGES = [
  { id: "overview", title: "Overview", icon: "dashboard" },
  { id: "universal", title: "Universal Restoration", icon: "auto_fix_high" },
  { id: "hard-routed", title: "Hard-Routed Restoration", icon: "alt_route" },
  { id: "soft-moe", title: "Soft Mixture-of-Experts Restoration", icon: "hub" },
  { id: "face-to-sketch", title: "Face-to-Sketch Generator", icon: "draw" },
];

// ONNX files each workspace needs (names match backend/app/models.py).
export const REQUIRED_MODELS = {
  universal: ["task1_udae.onnx"],
  "hard-routed": [
    "task2_classifier.onnx",
    "task2_specialist_salt_pepper.onnx",
    "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx",
  ],
  "soft-moe": ["task3_moe.onnx"],
  "face-to-sketch": ["task4_generator.onnx"],
};

// Accepted upload types (checked in the browser and again by the backend).
export const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
export const UNSUPPORTED_MESSAGE = "Unsupported file type. Upload a PNG or JPG image.";

// Format a time in milliseconds, e.g. 38.2 -> "38.2 ms".
export function formatMs(ms) {
  if (ms === null || ms === undefined) return "—";
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms`;
}
