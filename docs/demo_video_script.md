# Demo video script (about 6 minutes)

**What the brief requires the video to show:** the application startup process, image uploading, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, and experiment-tracking records. Length 5 to 7 minutes. Upload to YouTube (Unlisted is fine) and put only the link in the report.

The checklist at the end ticks off each requirement against the scene that covers it.

## Before you press record

1. **Recording tool:** use the Windows **Snipping Tool's video mode** (`Win + Shift + R`, then drag over the whole screen and press Start) or OBS Studio. Do **not** use the Xbox Game Bar (`Win + Alt + R`): it records only one window, so switching between the terminal and the browser would not be captured.
2. **Prepare files:** one pet photo of your own (not from the dataset), one face photo (`person2.jpeg` worked well), and nothing else open on screen.
3. **Browser tabs, already open:** the W&B project (https://wandb.ai/aleenababar04-fast-nuces/genai-a1) and http://localhost:8080 (it will show "offline" until you start the app; that is fine).
4. **Stop the app** so you can show it starting: open PowerShell in the project folder and run `docker compose down`. Keep that window open.
5. Close notifications (Windows Focus, `Win + N` then turn on Do not disturb).
6. Read the script aloud once. At a normal pace it is about 6 minutes.

## The script

Each scene has **Show** (what is on screen) and **Say** (read it, or say it in your own words). Times are cumulative.

### Scene 1: Introduction (0:00 to 0:20)

**Show:** the browser on the W&B project page, or a blank desktop.

**Say:**
"Hello, I'm Aleena Babar, roll number 23I-0628. This is my Generative AI Assignment 1: four image systems, three that restore damaged pet photos and one that turns face photos into sketches, all served from one web application. I'll show the experiment tracking first, then start the application and run all four tasks."

### Scene 2: Experiment tracking in Weights & Biases (0:20 to 1:30)

**Show:** the W&B project page with the list of runs, grouped. Click one group to expand it.

**Say:**
"Every training run, every hyperparameter-search trial and every evaluation was recorded in Weights & Biases. There are 83 runs, grouped by task: the universal autoencoder, the classifier, the specialist autoencoders, the mixture of experts, and the GAN."

**Show:** inside the `task1-udae` group, point at the `trial-...` runs.

**Say:**
"These are the Optuna trials. Optuna tried different learning rates, bottleneck sizes, dropout and loss weights. Each trial is a short training run, and weak trials were stopped early. The best settings were then used for a full training run."

**Show:** open `final-task1_udae`, the **Charts** tab (training loss and validation curves going down/up).

**Say:**
"This is the final training of the universal autoencoder: the training loss falls and the validation quality rises steadily over 80 epochs, without overfitting."

**Show:** open `final-task4_cgan` and scroll to the **samples** images (the same faces logged every 10 epochs). Then the **Artifacts** tab.

**Say:**
"For the GAN I logged the same validation faces every 10 epochs, so you can see the sketches improving during training, and the four GAN losses are logged separately. Every final model is also saved here as an artifact."

### Scene 3: Application startup (1:30 to 2:15)

**Show:** the PowerShell window in the project folder. Type and run:

```
docker compose up --build -d
```

then

```
docker compose ps
```

**Say (while it starts):**
"The application runs in two Docker containers started with one command. The backend is a FastAPI server that loads the seven trained models, exported to ONNX format. The frontend is a React and Tailwind interface served by nginx, which forwards requests to the backend. The frontend only starts once the backend reports that it is healthy."

**Show:** `docker compose ps` shows both containers up and the backend healthy. Switch to the browser, refresh http://localhost:8080.

**Say:**
"Both containers are running. The status card shows the backend is online and all seven models are loaded."

(If the build takes long, pause the recording while you wait and resume when it finishes.)

### Scene 4: Overview (2:15 to 2:30)

**Show:** the Overview page, then point at the sidebar.

**Say:**
"The application has four workspaces, one for each task. I designed the interface first in Google Stitch and then built it in React."

### Scene 5: Universal Restoration (2:30 to 3:30)

**Show:** open **Universal Restoration**. **Upload your own pet photo** (drag it in). Choose **Salt-and-pepper**, **High**, press **Restore Tensor**.

**Say:**
"First, the universal denoising autoencoder. I'm uploading my own photo, which the model has never seen. I choose salt-and-pepper noise at high severity. The corruption is applied at runtime by the backend, then the model restores it."

**Show:** point at the three panels, then the settings chips and the inference time.

**Say:**
"On the left is the original, in the middle the corrupted image the model receives, and on the right the restored output. The noise is removed. The settings used, a noise probability of 0.15, and the inference time are shown below. One model handles every corruption without being told which one it is, so its outputs are slightly smooth: that comes from its compressed bottleneck."

**Show:** choose **Occlusion / Medium**, press Restore. Then press **Download result** and show the downloaded file appearing.

**Say:**
"With occlusion, black boxes are added and the model fills them in from the surrounding content. I can download the result as a PNG file."

### Scene 6: Hard-Routed Restoration (3:30 to 4:30)

**Show:** open **Hard-Routed Restoration**. Pick a sample (or your photo), choose **Gaussian blur / High**, press **Route & Restore**.

**Say:**
"The second system first classifies the corruption, then sends the image to one specialist autoencoder trained only for that corruption. Here the classifier gives blur almost one hundred percent probability, so the blur specialist is selected."

**Show:** point at the four probability bars, the predicted corruption and the selected expert.

**Say:**
"These are the four classifier probabilities: clean, salt-and-pepper, blur and occlusion. On the test set the classifier is about 99.6 percent accurate."

**Show:** choose **None**, press Route & Restore.

**Say:**
"If the image is clean, the classifier routes it to the identity bypass, and the image is returned unchanged instead of being processed by an expert."

### Scene 7: Soft Mixture-of-Experts (4:30 to 5:30)

**Show:** open **Soft Mixture-of-Experts Restoration**. Choose **Occlusion / High**, press **Blend & Restore**.

**Say:**
"The third system does not pick one expert. A gating network gives a weight to each of four branches: the original image, and the three specialists. The output is the weighted blend. Here almost all the weight goes to the occlusion expert, which is highlighted as the top contributor."

**Show:** choose **Gaussian blur / Low**, press Blend & Restore. Point at the stacked weight bar.

**Say:**
"With mild blur, the weight is shared between the blur expert and the original image. The gate learned this itself during training: for mild damage, keeping part of the original image gives a better result than full restoration. The four weights always add up to one."

### Scene 8: Face-to-Sketch Generator (5:30 to 6:20)

**Show:** open **Face-to-Sketch Generator**. Upload the face photo, choose **Style 1**, press **Generate**. Then **Style 2**, then **Style 3**.

**Say:**
"The last task is a conditional GAN that turns a face photo into a sketch. The style is a learned input to both the generator and the discriminator. Style 1 is a light outline, Style 2 has heavy dark shading, and Style 3 is a softer toned sketch, all from the same photo."

**Show:** (optional) switch to **Use webcam**, capture, generate. Then press **Download sketch**.

**Say:**
"It also works with the webcam, and the sketch can be downloaded."

### Scene 9: Close (6:20 to 6:35)

**Show:** the Overview page or the W&B project.

**Say:**
"To summarise: four models, tuned with Optuna, tracked in Weights & Biases, exported to ONNX and served in one Docker application. The code, configuration and instructions are in the GitHub repository linked in my report. Thank you."

## Checklist of what the brief requires

| Required in the brief | Scene |
|---|---|
| Application startup process | 3 |
| Image uploading | 5 (your own pet photo), 8 (face) |
| Runtime corruption | 5, 6, 7 (chosen corruption applied by the backend) |
| Universal restoration | 5 |
| Hard routing | 6 |
| Soft expert weights | 7 |
| Face-to-sketch generation | 8 |
| Result downloading | 5 and 8 |
| Experiment-tracking records | 2 |

## After recording

1. Trim the start and end if needed (Clipchamp, which comes with Windows, is enough).
2. Upload to YouTube: **Create → Upload video**, visibility **Unlisted**.
3. Copy the link (`https://youtu.be/...`) and send it to Claude Code so it can be added to the report and the README.
