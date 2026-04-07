import numpy as np
from PIL import Image
import random
import torch
from typing import Any, List

def visualize_masks(image_path: str, masks: List[torch.Tensor], output_path: str, alpha: float = 1):
    """
    Overlay segmentation masks over the image.

    Args:
        image_path: path to original image
        masks: list of masks of shape (H, W)
        output_path: where to save visualization
        alpha: transparency of masks
    """

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32)

    overlay = image_np.copy()

    for i, mask in enumerate(masks):
        if mask is None:
            continue

        # 🔹 Convert torch.Tensor → numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Ensure binary mask
        mask = mask > 0

        # Generate deterministic color (better than random for debugging)
        color = np.array([
            (i * 50) % 255,
            (i * 80) % 255,
            (i * 110) % 255
        ])

        # Apply mask
        overlay[mask] = (
            (1 - alpha) * overlay[mask] +
            alpha * color
        )

    overlay = overlay.astype(np.uint8)

    Image.fromarray(overlay).save(output_path)