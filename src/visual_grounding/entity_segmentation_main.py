# """
# The inputs file format should be like the following - 

# outputs (of extraction) - 
#     |________________________ img_folder 
#                                 |_____________ entities.jsonl (per caption entities and their respective counts)
#                                 |_____________ image.jpg 
#                                 |_____________ entities_masks_bboxes_confidence_levels.json
#                                 |_____________ caption_1_entites_masks_bboxes_cl.json

# visualization_outputs - 
#     |_____________________ img_folder
#                                 |____________ caption_1_entites_visualization.jpg
#                                 |____________ caption_2_entities_visualization.jpg

# """

# import os
# import logging
# import torch
# import numpy as np
# from PIL import Image
# from pathlib import Path

# from visual_grounding.models.sam3_model import Sam3_Segmentation
# from visual_grounding.config.config_loader import ConfigLoader

# logger = logging.getLogger(__name__)

# class EntitySegementation:

#     """
#     High-level pipeline for entity-wise segmentation using SAM3.

#     Responsibilities:
#     - Iterate over Stage 1 output folders
#     - Read entities.jsonl
#     - Run segmentation per entity
#     - Cache results to avoid recomputation
#     - Save per-caption and per-image outputs
#     """

#     def __init__(self, config_path: str) -> None:
        
#         self.config_loader = ConfigLoader(config_path)

#         self.input_folder = Path(self.config_loader.get_input_folder())
#         self.output_folder = Path(self.config_loader.get_output_folder())
#         self.prompt_path = Path(self.config_loader.get_prompt_path())

#         self.use_count_hint = self.config_loader.get_use_count_hint()

#         self.segmentor = Sam3_Segmentation(config_path)

#         logger.info(f"Input folder: {self.input_folder}")
#         logger.info(f"Output folder: {self.output_folder}")

    
#     def run(self):
#         """
#         Entry point to process all image folders.
#         """

#         for img_folder in iter(self.input_folder):
#             img_folder_path = os.path.join(self.input_folder, img_folder)

#             if not img_folder_path
        

