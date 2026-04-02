import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional

from transformers import Sam3Model, Sam3Processor

from visual_grounding.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class EntitySegementation:

    """
    Wraps SAM 3 (HuggingFace Transformers) for text-prompted instance segmentation.
    """