"""
NeuroLens — Interactive MRI classification demo.

A self-contained Gradio application that loads the trained checkpoint and
exposes the model through a polished web UI suitable for Hugging Face Spaces.

Deployment notes
----------------
Required files in the Space root:

* ``app.py``                     — this file
* ``stage2_best.pt``             — model checkpoint (~41 MB)
* ``examples/`` or ``samples/``  — folder of sample MRI images (optional)
* ``requirements.txt``           — dependencies

The app gracefully handles missing checkpoints and missing example folders,
so a partial upload during deployment will not crash the runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
INDEX_TO_LABEL = dict(enumerate(CLASS_LABELS))
NUM_CLASSES = len(CLASS_LABELS)

INPUT_RESOLUTION = 224
RESIZE_TARGET = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CHECKPOINT_PATH = Path("stage2_best.pt")
# Support either an "examples" or a "samples" folder (whichever exists).
EXAMPLES_DIR = Path("examples") if Path("examples").exists() else Path("samples")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NeuroLens] Active compute device: {DEVICE}")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model() -> torch.nn.Module:
    """Construct EfficientNet-B3 and restore weights from ``stage2_best.pt``."""
    print(f"[NeuroLens] Loading checkpoint from {CHECKPOINT_PATH}...")
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    state_dict = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print("[NeuroLens] Model ready.")
    return model


MODEL = load_model()
CAM_ENGINE = GradCAM(model=MODEL, target_layers=[MODEL.blocks[-1]])


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(RESIZE_TARGET),
        transforms.CenterCrop(INPUT_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

DISPLAY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(RESIZE_TARGET),
        transforms.CenterCrop(INPUT_RESOLUTION),
        transforms.ToTensor(),
    ]
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def coerce_to_pil(image) -> Image.Image:
    """Normalise a Gradio input (numpy / PIL / RGBA / grayscale) into RGB PIL."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def confidence_band(confidence: float) -> tuple[str, str]:
    """Map a confidence value to a (label, emoji) pair for the UI summary."""
    if confidence >= 0.90:
        return "High confidence", "🟢"
    if confidence >= 0.70:
        return "Moderate confidence", "🟡"
    return "Low confidence — interpret with caution", "🟠"


def empty_response() -> tuple[dict, np.ndarray, str]:
    """Return a neutral placeholder triple when no image has been provided."""
    blank = np.zeros((INPUT_RESOLUTION, INPUT_RESOLUTION, 3), dtype=np.uint8)
    return (
        {name: 0.0 for name in CLASS_LABELS},
        blank,
        "**Upload a brain MRI image or pick one of the examples below to begin.**",
    )


# --------------------------------------------------------------------------- #
# Inference handler
# --------------------------------------------------------------------------- #

def classify(input_image) -> tuple[dict, np.ndarray, str]:
    """Top-level Gradio handler — predict, render Grad-CAM, format summary."""
    if input_image is None:
        return empty_response()

    pil_image = coerce_to_pil(input_image)

    # ------------------------ Forward pass ------------------------ #
    input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    rgb_image = DISPLAY_TRANSFORM(pil_image).permute(1, 2, 0).numpy()

    with torch.no_grad():
        logits = MODEL(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_index = int(probabilities.argmax())
    predicted_label = INDEX_TO_LABEL[predicted_index]
    confidence = float(probabilities[predicted_index])

    # ------------------------ Grad-CAM ----------------------------- #
    targets = [ClassifierOutputTarget(predicted_index)]
    heatmap = CAM_ENGINE(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)

    # ------------------------ Output shape ------------------------- #
    probability_distribution = {
        INDEX_TO_LABEL[i]: float(probabilities[i]) for i in range(NUM_CLASSES)
    }

    sorted_indices = np.argsort(probabilities)[::-1]
    runner_up_label = INDEX_TO_LABEL[sorted_indices[1]]
    runner_up_prob = probabilities[sorted_indices[1]]

    band_label, band_emoji = confidence_band(confidence)

    summary = f"""
### Prediction: **{predicted_label.upper()}**

**Confidence:** {confidence:.1%} {band_emoji} {band_label}

**Next most likely:** {runner_up_label} ({runner_up_prob:.1%})

The heatmap on the right highlights the regions that most influenced this prediction.
Warm colours (red / yellow) indicate areas of strong model attention; cool colours
(blue / green) indicate areas the model essentially ignored.
"""

    return probability_distribution, overlay, summary


# --------------------------------------------------------------------------- #
# UI copy
# --------------------------------------------------------------------------- #

HEADER_MARKDOWN = """
# 🧠 NeuroLens — Brain MRI Tumor Classifier

> ⚠️ **Medical disclaimer:**
> NeuroLens is a research and portfolio demonstration. It has **not** been
> clinically validated, has **not** been reviewed by qualified radiologists,
> and **must not** be used for any medical or diagnostic decision-making.
> Always consult a licensed healthcare professional for medical concerns.

This demo classifies a brain MRI slice into one of four categories —
**glioma**, **meningioma**, **pituitary tumor**, or **no tumor** — and
produces a **Grad-CAM attention map** showing which regions of the image
influenced the prediction.

**How to use:** Upload your own MRI image or pick one of the bundled examples
below. Results appear instantly.

**Under the hood:** EfficientNet-B3 trained with strict patient-level splitting
to avoid the data leakage that inflates results in most public implementations.
Test accuracy: **96.51%** across 687 unseen patients · macro AUC **0.9977**.
"""

ARTICLE_MARKDOWN = """
### Why patient-level splitting matters

Most public brain-tumor classifier notebooks report 96–99% accuracy on this
dataset. **Many of those numbers are inflated by data leakage.** The Figshare
source contains MRI slices from only 233 unique patients, with each patient
contributing roughly 13 slices on average. Splitting at the *image* level
puts multiple slices from the same patient in both the training and test sets,
so the model memorises patient-specific anatomy rather than learning tumor
features.

NeuroLens implements **patient-level splitting** verified with explicit
set-intersection checks, ensuring the model is evaluated on patients it has
genuinely never seen.

### Known limitations

* Meningioma recall is ~85% — this is the model's hardest class and is
  flagged transparently in the project's failure analysis.
* The model uses a single MRI modality (T1-weighted). Multi-modal inputs
  (T1, T2, FLAIR, T1ce) would likely improve meningioma vs glioma
  separation.
* Training data was collected from specific hospitals; cross-scanner
  generalisation has not been validated.

Built with PyTorch · `timm` · `pytorch-grad-cam` · Gradio.
"""


# --------------------------------------------------------------------------- #
# Example gallery
# --------------------------------------------------------------------------- #

def discover_examples() -> Optional[list[list[str]]]:
    """Return a list of example image paths if an examples/samples folder exists."""
    if not EXAMPLES_DIR.exists():
        return None
    candidates = sorted(
        str(p)
        for p in EXAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return [[path] for path in candidates] if candidates else None


example_gallery = discover_examples()
print(f"[NeuroLens] Example images discovered: "
      f"{len(example_gallery) if example_gallery else 0}")


# --------------------------------------------------------------------------- #
# Assemble interface
# --------------------------------------------------------------------------- #

custom_css = """
.gradio-container h1 { text-align: center; }
.gradio-container { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
footer { display: none !important; }
"""

interface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Brain MRI image", height=360),
    outputs=[
        gr.Label(
            num_top_classes=NUM_CLASSES,
            label="Predicted probability per class",
        ),
        gr.Image(
            type="numpy",
            label="Grad-CAM attention overlay",
            height=360,
        ),
        gr.Markdown(label="Prediction summary"),
    ],
    title="NeuroLens",
    description=HEADER_MARKDOWN,
    article=ARTICLE_MARKDOWN,
    examples=example_gallery,
    cache_examples=False,
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"),
    flagging_mode="never",
)


if __name__ == "__main__":
    interface.launch()
