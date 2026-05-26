"""
NeuroLens — Model architecture.

Defines the convolutional backbone used for 4-way brain MRI classification.
The architecture is an EfficientNet-B3 (~10.7M parameters) with a custom
4-class linear head, loaded via the `timm` library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import timm
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BACKBONE_NAME: str = "efficientnet_b3"
NUM_CLASSES: int = 4

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def create_classifier(pretrained: bool = False) -> nn.Module:
    """Instantiate the EfficientNet-B3 backbone with a 4-class linear head.

    Args:
        pretrained: When ``True``, ImageNet pretrained weights are loaded
            from ``timm``. Use ``False`` when restoring our own checkpoint.

    Returns:
        A ``torch.nn.Module`` ready for training or evaluation.
    """
    return timm.create_model(
        BACKBONE_NAME,
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )


def restore_from_checkpoint(
    checkpoint_path: PathLike,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Build the model and load trained weights from a state-dict file.

    Args:
        checkpoint_path: Path to the ``.pt`` file containing a state dictionary.
        device: Target device for the loaded model (``"cpu"`` or ``"cuda"``).

    Returns:
        Model placed on ``device`` and switched to ``eval`` mode.
    """
    model = create_classifier(pretrained=False)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the convolutional layer used as the Grad-CAM target.

    For EfficientNet-B3 we target the final block, which captures the
    highest-level semantic features before global pooling.
    """
    return model.blocks[-1]
