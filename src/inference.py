"""
NeuroLens — Inference and Grad-CAM utilities.

This module wraps the trained model in a small, focused API:

* :func:`run_prediction` produces class probabilities for a single image.
* :func:`render_attention_map` produces a Grad-CAM heatmap overlay.

The functions are intentionally side-effect free so they can be reused
from the Gradio app, batch evaluation scripts, or external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .dataset import (
    CLASS_LABELS,
    INDEX_TO_LABEL,
    build_evaluation_transform,
    build_visualisation_transform,
)

# Module-level transforms — built once, reused across calls.
_EVAL_TRANSFORM = build_evaluation_transform()
_VIS_TRANSFORM = build_visualisation_transform()


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PredictionResult:
    """Structured result returned by :func:`run_prediction`."""

    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    runner_up_label: str
    runner_up_confidence: float


# --------------------------------------------------------------------------- #
# Core inference
# --------------------------------------------------------------------------- #

def run_prediction(
    model: nn.Module,
    image: Image.Image,
    device: str | torch.device = "cpu",
) -> PredictionResult:
    """Run a forward pass and return a structured prediction.

    Args:
        model: Trained classification model (expected to be in ``eval`` mode).
        image: A PIL image — automatically converted to RGB.
        device: Device on which to run inference.

    Returns:
        A :class:`PredictionResult` containing the top-1 label, its
        confidence, the full probability distribution, and the second-most
        likely label.
    """
    image = image.convert("RGB")
    input_tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    sorted_indices = np.argsort(probabilities)[::-1]
    top_index = int(sorted_indices[0])
    runner_up_index = int(sorted_indices[1])

    return PredictionResult(
        predicted_label=INDEX_TO_LABEL[top_index],
        confidence=float(probabilities[top_index]),
        probabilities={
            INDEX_TO_LABEL[i]: float(probabilities[i])
            for i in range(len(CLASS_LABELS))
        },
        runner_up_label=INDEX_TO_LABEL[runner_up_index],
        runner_up_confidence=float(probabilities[runner_up_index]),
    )


# --------------------------------------------------------------------------- #
# Visual explanation (Grad-CAM)
# --------------------------------------------------------------------------- #

def render_attention_map(
    model: nn.Module,
    image: Image.Image,
    target_layer: nn.Module,
    device: str | torch.device = "cpu",
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Render a Grad-CAM attention overlay for a single image.

    Args:
        model: Trained classification model.
        image: Input PIL image.
        target_layer: Convolutional layer whose activations are used to
            compute the heatmap (typically the last conv block).
        device: Inference device.
        target_class: Class index to visualise. When ``None``, the model's
            top-1 prediction is used.

    Returns:
        ``(H, W, 3)`` ``uint8`` numpy array containing the heatmap composited
        onto the original RGB image.
    """
    image = image.convert("RGB")
    input_tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    rgb_image = _VIS_TRANSFORM(image).permute(1, 2, 0).numpy()

    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target_class = int(logits.argmax())

    cam = GradCAM(model=model, target_layers=[target_layer])
    heatmap = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]

    return show_cam_on_image(rgb_image, heatmap, use_rgb=True)
