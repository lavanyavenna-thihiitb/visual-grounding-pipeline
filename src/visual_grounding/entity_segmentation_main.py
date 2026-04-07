# import logging
# import torch
# import numpy as np
# from PIL import Image

# from visual_grounding.models.sam3_model import Sam3Segementation
# from visual_grounding.config.config_loader import ConfigLoader

# logger = logging.getLogger(__name__)

# class EntitySegementation:

#     """
#     Wraps SAM 3 (HuggingFace Transformers) for text-prompted instance segmentation.

#     Loads the SAM 3 model once in __init__ and exposes a single
#     segmentation() method that accepts an image path + caption and
#     returns a JSON-serialisable list of per-instance results.
#     """

#     def __init__(self, config_path: str) -> None:
        
#         #sam3 model
#         self.model = Sam3Segementation(config_path)
#         self.batch_size = self.model.config_loader.get_batch_size()

#     def segmentation(self, image_path: str, caption: str) -> list[dict]:
#         """
#         Run SAM 3 segmentation for all entities in a caption on one image.

#         Flow:
#             1. Load image from image_path as PIL image
#             2. Given the dictionary from stage 1 for each image_path as the parameter, for each entity format a text prompt using YAML templates
#         """

        

