"""
NeuroLens — Dataset and image transforms.

Provides a PyTorch ``Dataset`` for the preprocessed MRI slices along with
the training and evaluation augmentation pipelines.

Class index convention (kept consistent across the project)::

    0 = glioma
    1 = meningioma
    2 = notumor
    3 = pituitary
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Class label registry
# --------------------------------------------------------------------------- #

CLASS_LABELS: Sequence[str] = ("glioma", "meningioma", "notumor", "pituitary")

LABEL_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_LABELS)}

# --------------------------------------------------------------------------- #
# Image normalisation constants
# --------------------------------------------------------------------------- #

INPUT_RESOLUTION: int = 224
_RESIZE_TARGET: int = 256

# Standard ImageNet statistics — used because the backbone was pretrained on ImageNet.
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class MRISliceDataset(Dataset):
    """A ``torch.utils.data.Dataset`` over preprocessed brain MRI slices.

    The constructor expects a ``pandas.DataFrame`` containing, at minimum:

    * ``processed_path`` — filesystem path to a preprocessed PNG image
    * ``label`` — one of the four strings in :data:`CLASS_LABELS`
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self._manifest = manifest.reset_index(drop=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self._manifest.iloc[index]

        image = Image.open(row["processed_path"]).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)

        target = LABEL_TO_INDEX[row["label"]]
        return image, target


# --------------------------------------------------------------------------- #
# Transform factories
# --------------------------------------------------------------------------- #

def build_training_transform() -> transforms.Compose:
    """Augmentation pipeline used during training.

    Notably, **no horizontal flip is applied** — the human brain is
    anatomically asymmetric, so flipping would corrupt the lateral cues
    the model relies on for localisation.
    """
    return transforms.Compose(
        [
            transforms.Resize(_RESIZE_TARGET),
            transforms.CenterCrop(INPUT_RESOLUTION),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_evaluation_transform() -> transforms.Compose:
    """Deterministic transform used at validation, test, and inference time."""
    return transforms.Compose(
        [
            transforms.Resize(_RESIZE_TARGET),
            transforms.CenterCrop(INPUT_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_visualisation_transform() -> transforms.Compose:
    """Transform that produces an un-normalised tensor for overlay rendering.

    Used by the Grad-CAM pipeline so the heatmap can be composited onto a
    correctly-coloured RGB image.
    """
    return transforms.Compose(
        [
            transforms.Resize(_RESIZE_TARGET),
            transforms.CenterCrop(INPUT_RESOLUTION),
            transforms.ToTensor(),
        ]
    )
