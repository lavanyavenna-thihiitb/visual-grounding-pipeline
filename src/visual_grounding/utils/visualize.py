"""
Visualization Module for Entity Segmentation Results
 
This module provides visualization utilities for segmentation results:
1. visualize_masks_for_output_directory: Overlays colored masks with entity labels
2. visualize_bboxes_for_output_directory: Draws bounding boxes with entity labels and scores
 
Output files:
    - {caption_type}_mask_visualization.jpg: Colored mask overlays with labels
    - {caption_type}_bbox_visualization.jpg: Bounding boxes with labels and confidence scores
"""


import numpy as np
from PIL import Image
import random
import torch
from pathlib import Path
from typing import Any, List
import logging
import json

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
 

def visualize_masks_for_output_directory(output_path: Path, logger: logging.Logger) -> None:

    """
    Creates visualizations with colored mask overlays and entity labels.
    
    Process:
    1. Load base image from output directory
    2. For each caption type JSON:
        a. Extract entities and their mask instances
        b. Create colored overlay for each entity
        c. Add entity labels at instance centroids
        d. Save visualization as {caption_type}_mask_visualization.jpg
    
    Output:
        - Colored masks with 50% opacity overlay
        - Entity labels positioned at mask centroid
        - Confidence scores displayed below entity name
        - One visualization per caption type
    
    Args:
        output_path (Path): Path to image output directory containing:
            - {folder_name}.jpg (base image)
            - *_entities.json (caption files)
            - masks/ (directory with .npy mask files)
        logger (logging.Logger): Logger instance
        
    Returns:
        None
        
    Logs:
        - info: Successful visualization saves
        - warning: Missing images or entity files
        - error: Failed loads or processing
    """
    
    # Retrieving the image that is stored as - output_path.jpg, folder name == image name
    image_path = output_path / f"{output_path.name}.jpg"

    if not image_path:
        logger.warning(f"No image found with name {image_path}, maybe foldername is not equal to image name")
        logger.warning(f"Visualization of image is not possible.")
        return
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Find all list of .json files for captions
    json_files = list(output_path.glob("*_entities.json"))

    if not json_files:
        logger.warning(f"No entity json files found in {output_path}")
        return
    
    #Iterate over json files
    for caption_json_file in json_files:

        #Open file
        with open(caption_json_file, "r") as f:
            data = json.load(f)

        visualize_image = image_np.copy()

        #Prepare overlay
        overlay_image = visualize_image.copy()

        for entity_data in data["entities"]:
            entity_name = entity_data["entity"]

            # Random color per entity
            color = [random.randint(0, 255) for _ in range(3)]

            for instance in entity_data["instances"]:
                mask_path = output_path / instance["mask_path"]

                if not mask_path.exists():
                    continue

                # Load mask path using numpy
                mask = np.load(mask_path)

                # Ensure binary mask
                mask = (mask > 0).astype(np.uint8)

                # Apply color overlay
                for c in range(3):
                    overlay_image[:, :, c] = np.where(
                        mask == 1,
                        overlay_image[:, :, c] * 0.5 + color[c] * 0.5,
                        overlay_image[:, :, c]
                    )

        # Save visualization
        output_file = output_path / f"{data['caption_type']}_visualization.jpg"
        Image.fromarray(overlay_image.astype(np.uint8)).save(output_file)

        logger.info(f"Saved visualization: {output_file}")


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