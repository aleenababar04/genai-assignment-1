# Google Stitch prompt pack — RestoreLab

Prompts for designing the assignment web app in Google Stitch (stitch.withgoogle.com) before it is built in React + Tailwind CSS.

Product name used throughout: **RestoreLab**.

Exact workspace names (do not let Stitch rename them):

1. Universal Restoration
2. Hard-Routed Restoration
3. Soft Mixture-of-Experts Restoration
4. Face-to-Sketch Generator

---

## 1. How to use this file

**Before you start**

- Sign in to Stitch and start a new project. Name it "RestoreLab".
- Choose the web / desktop option (not mobile app) before generating. The app is desktop-first.
- Note: Stitch's interface changes often and this file was written without checking the current version. Where a step below says "export or screenshot", use whatever the current interface offers. If a button named here does not exist, the action still applies: get an image of the screen and keep the prompt you used.

**Order**

Paste one prompt at a time, in this order. One prompt produces one screen. Wait for each screen to finish before pasting the next.

| Step | Prompt | Screen |
|---|---|---|
| 0 | Prompt 0 | Design system (style guide) |
| 1 | Prompt 1 | Overview |
| 2 | Prompt 2 | Universal Restoration |
| 3 | Prompt 3 | Hard-Routed Restoration |
| 4 | Prompt 4 | Soft Mixture-of-Experts Restoration |
| 5 | Prompt 5 | Face-to-Sketch Generator |
| 6 | Prompt 6 | Face-to-Sketch Generator, webcam capture state |
| 7 | Prompt 7 | Loading state |
| 8 | Prompt 8 | Error and empty state |

Copy the whole contents of the code block, nothing outside it. Every prompt repeats the app name, style and navigation on purpose, because Stitch may not carry context from one prompt to the next.

**How to iterate**

- After each screen, check it against the Requirements checklist (section 5) before moving on.
- If something is wrong, ask for one change at a time in a follow-up message on that screen, for example "Rename the page title to exactly 'Hard-Routed Restoration'". Several changes in one message tend to cause unrelated parts to be redrawn.
- If a screen is badly off, regenerate from the original prompt rather than stacking many fixes.
- Section 4 has ready-made follow-up prompts for common problems.
- If Stitch has a limit on generations per day, do Prompts 2 to 5 first; they carry the graded requirements.

**What to save as evidence for the report**

Suggested folder: `report/figures/stitch/`

| What | Suggested file name |
|---|---|
| Design system screen | `00_design_system.png` |
| Overview | `01_overview.png` |
| Universal Restoration | `02_universal_restoration.png` |
| Hard-Routed Restoration | `03_hard_routed_restoration.png` |
| Soft Mixture-of-Experts Restoration | `04_soft_moe_restoration.png` |
| Face-to-Sketch Generator | `05_face_to_sketch.png` |
| Webcam capture state | `06_face_to_sketch_webcam.png` |
| Loading state | `07_loading_state.png` |
| Error and empty state | `08_error_state.png` |
| Any responsive or dark variants | `09_mobile_<workspace>.png`, `10_dark_<workspace>.png` |
| A screenshot of the whole Stitch project canvas showing all screens together | `stitch_project_canvas.png` |
| The Stitch project link | `stitch_project_link.txt` |
| Any export Stitch offers (HTML/CSS, Figma) | `export/` subfolder |

Also keep:

- The prompts actually used. If you edit a prompt before pasting, save the edited text (for example in `prompts_used.md` in the same folder), including the follow-up prompts.
- One screenshot that shows a prompt and its generated screen together inside the Stitch interface, with the date visible if possible. This shows the design was made in Stitch and not drawn afterwards.

---

## 2. Prompt 0 — design system

**Direction chosen:** a neutral light interface (slate greys, white cards, one indigo accent) with four fixed category colours.

**Why it suits this tool:** the content that matters is the images and the numbers, so the chrome stays grey and quiet and colour is reserved for meaning: each corruption/expert category has one colour that is reused in every bar, badge and expert card, so the classifier probabilities in workspace 2 and the routing weights in workspace 3 read the same way. Every value is a default Tailwind colour, so the implementation needs no custom palette.

| Role | Hex | Tailwind name |
|---|---|---|
| Page background | `#F8FAFC` | slate-50 |
| Card surface | `#FFFFFF` | white |
| Border | `#E2E8F0` | slate-200 |
| Text | `#0F172A` | slate-900 |
| Secondary text | `#64748B` | slate-500 |
| Primary (buttons, active nav) | `#4F46E5` | indigo-600 |
| Category: Clean / Identity | `#10B981` | emerald-500 |
| Category: Salt-and-pepper | `#F59E0B` | amber-500 |
| Category: Blur | `#06B6D4` | cyan-500 |
| Category: Occlusion | `#D946EF` | fuchsia-500 |
| Status online | `#16A34A` | green-600 |
| Status offline / error | `#DC2626` | red-600 |
| Dark variant: background / card / border / text | `#0F172A` / `#1E293B` / `#334155` / `#F1F5F9` | slate-900 / 800 / 700 / 100 |

```text
Design a style guide screen for "RestoreLab", a desktop web application for image restoration and face-to-sketch generation using small 128x128 images. Web layout, light theme.

Show these sections as cards on one page:

Colours: page background #F8FAFC, card surface #FFFFFF, border #E2E8F0, text #0F172A, secondary text #64748B, primary indigo #4F46E5, status online #16A34A, error #DC2626. Four category colours shown as labelled swatches: Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.

Typography: Inter. Page title 24px semibold, card title 16px semibold, body 14px, caption 12px. Monospace font for numeric values such as "38 ms" and "0.08".

Spacing and cards: 8px spacing grid, 24px gaps between cards. Cards are white with a 1px border, 12px corner radius, a very soft shadow and 24px padding. No gradients, no glass effects.

Components: primary button (indigo), secondary button (white with border), disabled button, segmented control, dashed drag-and-drop upload zone, category badges in the four category colours, a horizontal percentage bar in each category colour, a stat tile showing "Inference time 38 ms", a status dot with "Backend online", and a square image panel (288px) with a thin border and a caption below.

Add a small dark-theme sample card: background #0F172A, card #1E293B, border #334155, text #F1F5F9, same accent and category colours.

Keep everything flat, simple and easy to build with Tailwind CSS.
```

---

## 3. Screen prompts

### Prompt 1 — Overview

```text
Desktop web app screen: the Overview page of "RestoreLab", one application with four image-model workspaces working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary colour indigo #4F46E5. Inter font, monospace for numeric values. Category colours: Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Overview highlighted. At the sidebar bottom a compact "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 38 ms".

Main area: page title "Overview" and one sentence: "Four workspaces for restoring corrupted images and generating sketches from face photos."

Below, a 2x2 grid of workspace cards. Each card has a small icon, the workspace name, a one-line description and an "Open workspace" button:
1. Universal Restoration - "One autoencoder restores any corruption."
2. Hard-Routed Restoration - "A classifier detects the corruption and routes the image to one specialist."
3. Soft Mixture-of-Experts Restoration - "A gating network blends four expert branches."
4. Face-to-Sketch Generator - "A conditional GAN turns a face photo into a sketch in three styles."

Under the grid, a full-width "System" card: backend status "Online", a row of four model chips each marked "Loaded", and "Last inference: 38 ms".
```

### Prompt 2 — Universal Restoration

```text
Desktop web app screen: the "Universal Restoration" workspace of "RestoreLab", an image-restoration tool working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values. Category colours for badges: Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Universal Restoration highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 38 ms".

Page title "Universal Restoration", subtitle "One autoencoder restores any corruption." Main area in two columns.

Column 1, "Input" card (340px wide): dashed drag-and-drop zone "Drop an image here or browse", note "PNG or JPG"; "Or pick a sample" row of six small square thumbnails; "Corruption type" as four option buttons in a 2x2 grid: None, Salt-and-pepper noise, Gaussian blur, Rectangular occlusion (Salt-and-pepper noise selected); "Severity" segmented control: Low, Medium, High (Medium selected); full-width indigo "Restore" button.

Column 2, "Result" card: two square image panels side by side, 288px each, crisp pixelated upscaling, labelled "Input image" and "Restored output", caption "128 x 128". Below, a "Corruption settings" row of chips: "Type: Salt-and-pepper noise", "Severity: Medium", "Noise probability: 0.08". A stat tile "Inference time 38 ms". A "Download result" button.
```

### Prompt 3 — Hard-Routed Restoration

```text
Desktop web app screen: the "Hard-Routed Restoration" workspace of "RestoreLab", an image-restoration tool working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values. Category colours: Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Hard-Routed Restoration highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 46 ms".

Page title "Hard-Routed Restoration", subtitle "A classifier predicts the corruption and routes the image to one specialist." Two columns.

Column 1, "Input" card (340px wide): dashed drag-and-drop zone "Drop an image here or browse", note "PNG or JPG"; "Or pick a sample" row of six small square thumbnails; "Corruption type" as four option buttons in a 2x2 grid: None, Salt-and-pepper noise, Gaussian blur, Rectangular occlusion (Gaussian blur selected); "Severity" segmented control: Low, Medium, High (Medium selected); full-width indigo "Restore" button.

Column 2, two stacked cards.
"Classifier routing" card: four labelled horizontal bars in the category colours with percentages: Clean 2.1%, Salt-and-pepper 3.4%, Blur 91.8%, Occlusion 2.7%. Below: "Predicted corruption: Blur" badge and "Selected expert: Blur expert", with a grey note "Identity bypass is used when the image is predicted clean."
"Result" card: two 288px square panels, pixelated upscaling, "Input image" and "Reconstructed output"; chips "Gaussian blur", "Medium", "Kernel 5", "Sigma 1.5"; stat tile "Inference time 46 ms"; "Download result" button.
```

### Prompt 4 — Soft Mixture-of-Experts Restoration

```text
Desktop web app screen: the "Soft Mixture-of-Experts Restoration" workspace of "RestoreLab", an image-restoration tool working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values. Category colours: Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Soft Mixture-of-Experts Restoration highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 61 ms".

Page title "Soft Mixture-of-Experts Restoration", subtitle "A gating network blends four branches." Two columns.

Column 1, "Input" card (340px wide): dashed drag-and-drop zone "Drop an image here or browse", note "PNG or JPG"; "Or pick a sample" row of six small square thumbnails; "Corruption type" as four option buttons in a 2x2 grid: None, Salt-and-pepper noise, Gaussian blur, Rectangular occlusion (Rectangular occlusion selected); "Severity" segmented control: Low, Medium, High (Medium selected); full-width indigo "Restore" button.

Column 2, two stacked cards.
"Routing weights" card: one stacked horizontal contribution bar with four segments in the category colours, then four expert cards in a row, each with name and weight: Identity / clean branch 0.06, Salt-and-pepper expert 0.09, Blur expert 0.11, Occlusion expert 0.74. The Occlusion expert card is highlighted with a coloured border and a "Top contributor" badge; low-weight cards look muted.
"Result" card: two 288px square panels, pixelated upscaling, "Input image" and "Reconstructed output"; chips "Rectangular occlusion", "Medium", "Occlusion 20%", "2 rectangles"; stat tile "Inference time 61 ms"; "Download result" button.
```

### Prompt 5 — Face-to-Sketch Generator

```text
Desktop web app screen: the "Face-to-Sketch Generator" workspace of "RestoreLab", a web application with four image-model workspaces working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Face-to-Sketch Generator highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 84 ms".

Page title "Face-to-Sketch Generator", subtitle "A conditional GAN turns a face photo into a sketch." Two columns.

Column 1, "Input" card (340px wide): a two-tab switch "Upload photo" (active) and "Use webcam". Under it a dashed drag-and-drop zone "Drop a face photo here or browse", note "PNG or JPG", showing the uploaded file name "portrait.jpg" with a small thumbnail. Then "Sketch style": three selectable style cards in a row labelled "Style 1", "Style 2", "Style 3", each with a small sketch preview, Style 2 selected with an indigo border. Then a full-width indigo "Generate sketch" button. There is no text prompt field anywhere.

Column 2, "Result" card: two square image panels side by side, 288px each, crisp upscaling, labelled "Original photograph" and "Generated sketch" (a pencil-style sketch of the same face), caption "128 x 128". Below: chip "Style 2", stat tile "Inference time 84 ms", and a "Download sketch" button.
```

### Prompt 6 — Face-to-Sketch Generator, webcam capture state

```text
Desktop web app screen: the "Face-to-Sketch Generator" workspace of "RestoreLab" in its webcam capture state, before any sketch has been generated. RestoreLab is a web application with four image-model workspaces working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Face-to-Sketch Generator highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 84 ms".

Page title "Face-to-Sketch Generator", subtitle "A conditional GAN turns a face photo into a sketch." Two columns.

Column 1, "Input" card (340px wide): a two-tab switch "Upload photo" and "Use webcam" (active). Below, a square live preview frame showing a person's face from a webcam, with a small red "Live" badge in the top-left corner and a faint centred square crop guide. Under the frame: an indigo "Capture" button with a camera icon and a secondary "Stop camera" button. Then "Sketch style": three selectable style cards in a row labelled "Style 1", "Style 2", "Style 3", Style 1 selected. Then a full-width "Generate sketch" button shown disabled. No text prompt field.

Column 2, "Result" card in an empty state: two 288px square placeholder panels with dashed borders labelled "Original photograph" and "Generated sketch", and centred grey text "Capture a photo and choose a style to generate a sketch." The "Download sketch" button is disabled and the inference time shows a dash.
```

### Prompt 7 — Loading state

```text
Desktop web app screen: the "Universal Restoration" workspace of "RestoreLab" in its loading state, while the model is running. RestoreLab is an image-restoration tool working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5. Inter font, monospace for numeric values.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Universal Restoration highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: 38 ms".

Page title "Universal Restoration", subtitle "One autoencoder restores any corruption." Two columns.

Column 1, "Input" card (340px wide): dashed drag-and-drop zone showing the uploaded file "sample_03.png", note "PNG or JPG"; "Or pick a sample" row of six small square thumbnails; "Corruption type" as four option buttons in a 2x2 grid: None, Salt-and-pepper noise, Gaussian blur, Rectangular occlusion (Gaussian blur selected); "Severity" segmented control: Low, Medium, High (High selected). The full-width "Restore" button is disabled and shows a small spinner with the text "Restoring...". All input controls look disabled.

Column 2, "Result" card: left 288px square panel "Input image" shows the blurred input. The right panel "Restored output" is a grey skeleton placeholder with a centred spinner and the text "Running model...". The corruption settings chips are visible: "Gaussian blur", "High". The "Inference time" stat tile shows a grey skeleton bar instead of a number. The "Download result" button is disabled.
```

### Prompt 8 — Error and empty state

```text
Desktop web app screen: the "Hard-Routed Restoration" workspace of "RestoreLab" showing an upload error, with no result yet. RestoreLab is an image-restoration tool working on 128x128 images.

Style: clean light theme. Page background #F8FAFC; white cards with 1px #E2E8F0 border, 12px radius, soft shadow, 24px padding. Text #0F172A, secondary text #64748B, primary indigo #4F46E5, error red #DC2626. Inter font, monospace for numeric values.

Left sidebar, 264px, white: "RestoreLab" logo; nav links Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator, with Hard-Routed Restoration highlighted. At the bottom a "System status" card: green dot "Backend online", "ONNX models: 4 of 4 loaded", "Last inference: none yet".

Page title "Hard-Routed Restoration", subtitle "A classifier predicts the corruption and routes the image to one specialist." Two columns.

Column 1, "Input" card (340px wide): the dashed drag-and-drop zone has a red border and a pale red background, with a warning icon, the rejected file name "notes.pdf", the message "Unsupported file type. Upload a PNG or JPG image." in red, and a "Choose another file" link. Below: "Or pick a sample" row of six small square thumbnails; "Corruption type" as four option buttons in a 2x2 grid: None, Salt-and-pepper noise, Gaussian blur, Rectangular occlusion (None selected); "Severity" segmented control: Low, Medium, High, greyed out; full-width "Restore" button disabled.

Column 2, two stacked cards in empty states.
"Classifier routing" card: four empty grey bar tracks labelled Clean, Salt-and-pepper, Blur, Occlusion, each showing a dash instead of a percentage.
"Result" card: two 288px square placeholder panels with dashed borders labelled "Input image" and "Reconstructed output", and centred grey text "No result yet. Upload an image or pick a sample to begin." "Download result" disabled.
```

---

## 4. Refinement prompts

Paste one at a time as a follow-up on the screen that needs it.

**R1 — identical input panel across the three restoration workspaces**

```text
Make the "Input" card on this screen match the Universal Restoration screen exactly: same 340px width, same order (upload zone, sample thumbnails, corruption type 2x2 buttons, severity segmented control, Restore button), same labels and same spacing. Change nothing else.
```

**R2 — fix a renamed workspace**

```text
The workspace names must be exactly: "Universal Restoration", "Hard-Routed Restoration", "Soft Mixture-of-Experts Restoration", "Face-to-Sketch Generator". Correct the sidebar and the page title to use these exact names. Change nothing else.
```

**R3 — tighten spacing**

```text
Tighten the layout: use 16px gaps between cards and 20px card padding, and reduce empty vertical space so the whole workspace fits in a 1440x900 window without scrolling. Keep all elements and text.
```

**R4 — mobile / responsive variant**

```text
Create a mobile variant of this screen at 390px width. Replace the sidebar with a top bar containing the "RestoreLab" logo and a menu button that opens the four workspace links. Stack everything in one column in this order: input card, result images (stacked vertically, full width), settings chips, inference time, download button. Put the system status as a compact row under the top bar. Keep all text and values.
```

**R5 — dark mode variant**

```text
Create a dark theme variant of this screen: page background #0F172A, cards #1E293B, borders #334155, text #F1F5F9, secondary text #94A3B8. Keep the indigo primary and the four category colours (Clean #10B981, Salt-and-pepper #F59E0B, Blur #06B6D4, Occlusion #D946EF). Keep the layout and content unchanged.
```

**R6 — more legible probability bars**

```text
Make the four classifier probability bars easier to read: each bar on its own row with the label on the left, a 12px-tall bar on a light grey track in the middle, and the percentage right-aligned in monospace. Bars keep their category colours. Make the highest-probability row bold. Change nothing else.
```

**R7 — clearer expert contribution**

```text
Make the expert contributions clearer: show the stacked contribution bar at 24px height with the weight printed inside each segment that is wide enough, add a legend with the four branch names and colours under it, and give the highest-weight expert card a 2px coloured border and a "Top contributor" badge. Keep the four weights as they are.
```

**R8 — image panels**

```text
Make both image panels exactly the same size, 288px square, side by side with a 24px gap, each with a 1px border, a label above and the caption "128 x 128" below. Images are shown enlarged with sharp pixel edges, not smoothed. Change nothing else.
```

**R9 — backend offline variant**

```text
Create a variant of this screen where the backend is unavailable: the system status card shows a red dot with "Backend offline" and "ONNX models: not loaded"; a red-bordered banner at the top of the main area reads "Cannot reach the backend. Check that the server is running and try again." with a "Retry" button; the main action button is disabled. Keep everything else unchanged.
```

**R10 — top-tab navigation alternative**

```text
Replace the left sidebar with a top navigation bar: "RestoreLab" logo on the left, five tabs (Overview, Universal Restoration, Hard-Routed Restoration, Soft Mixture-of-Experts Restoration, Face-to-Sketch Generator) in the middle with the current one underlined in indigo, and a compact system status on the right showing "Backend online", "4 of 4 models", "38 ms". Keep the main content unchanged.
```

If you use R10, apply it to every screen so the navigation stays the same across the app.

---

## 5. Requirements checklist

Tick each row on the generated screen before moving to the next prompt.

### Shared (check on every screen)

| Required element | Done |
|---|---|
| Product name "RestoreLab" shown | [ ] |
| Persistent navigation lists all four workspaces with exact names | [ ] |
| Current workspace highlighted in the navigation | [ ] |
| System status: backend health (online/offline) | [ ] |
| System status: which ONNX models are loaded | [ ] |
| System status: last inference time | [ ] |
| Same colours, fonts and card style as the design system screen | [ ] |

### Overview

| Required element | Done |
|---|---|
| Four workspace cards with exact names and a short description each | [ ] |
| A way to open each workspace from its card | [ ] |

### Workspace 1 — Universal Restoration

| Required element | Done |
|---|---|
| Page title is exactly "Universal Restoration" | [ ] |
| Drag-and-drop upload zone with "PNG, JPG" noted | [ ] |
| Sample image gallery | [ ] |
| Corruption type selector: None / Salt-and-pepper noise / Gaussian blur / Rectangular occlusion | [ ] |
| Severity selector: Low / Medium / High | [ ] |
| Input image panel | [ ] |
| Restored output panel | [ ] |
| Selected corruption settings shown (e.g. noise probability 0.08) | [ ] |
| Inference time in ms | [ ] |
| Download button for the result | [ ] |

### Workspace 2 — Hard-Routed Restoration

| Required element | Done |
|---|---|
| Page title is exactly "Hard-Routed Restoration" | [ ] |
| Input panel identical to workspace 1 (upload, samples, corruption type, severity) | [ ] |
| Four classifier probabilities as labelled bars with percentages: Clean, Salt-and-pepper, Blur, Occlusion | [ ] |
| Bars use the four category colours | [ ] |
| Predicted corruption shown | [ ] |
| Selected expert shown, with "Identity bypass" for images predicted clean | [ ] |
| Reconstructed image panel | [ ] |
| Inference time in ms | [ ] |
| Download button | [ ] |

### Workspace 3 — Soft Mixture-of-Experts Restoration

| Required element | Done |
|---|---|
| Page title is exactly "Soft Mixture-of-Experts Restoration" | [ ] |
| Input panel identical to workspace 1 | [ ] |
| All four routing weights shown: Identity / clean branch, Salt-and-pepper expert, Blur expert, Occlusion expert | [ ] |
| Visual indication of which experts contributed most (stacked bar and/or highlighted expert cards) | [ ] |
| Same category colours as workspace 2 | [ ] |
| Reconstructed result panel | [ ] |
| Inference time in ms | [ ] |
| Download button | [ ] |

### Workspace 4 — Face-to-Sketch Generator

| Required element | Done |
|---|---|
| Page title is exactly "Face-to-Sketch Generator" | [ ] |
| Upload option for a facial photograph (PNG, JPG noted) | [ ] |
| Webcam option | [ ] |
| Style selection limited to Style 1, Style 2, Style 3 | [ ] |
| No free-text prompt box | [ ] |
| Generate button | [ ] |
| Original photograph and generated sketch side by side | [ ] |
| Inference time in ms | [ ] |
| Download button for the sketch | [ ] |

### Webcam capture state

| Required element | Done |
|---|---|
| Live preview frame | [ ] |
| Capture button | [ ] |
| Style 1 / Style 2 / Style 3 selector still visible | [ ] |
| Empty state for the result panels | [ ] |

### Loading, error and empty states

| Required element | Done |
|---|---|
| Loading: spinner or skeleton in the output panel while the model runs | [ ] |
| Loading: action button disabled or showing progress | [ ] |
| Error: "Unsupported file type" message on a bad upload, with accepted formats stated | [ ] |
| Empty: result area explains what to do when there is no result yet | [ ] |

### Evidence saved for the report

| Item | Done |
|---|---|
| Screenshot of every generated screen in `report/figures/stitch/` | [ ] |
| Prompts actually used, including follow-ups | [ ] |
| Stitch project link | [ ] |
| Screenshot of the Stitch interface showing a prompt with its generated screen | [ ] |
| Any export (HTML/CSS or Figma) | [ ] |
