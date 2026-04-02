import logging
import torch
import numpy as np
from PIL import Image
from typing import Any

from transformers import Sam3Model, Sam3Processor

from visual_grounding.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class Sam_Segementation:

    def __init__(self, config_path: str) -> None:
        self.config_loader = ConfigLoader(config_path)
        self.model_type = self.config_loader.get_model_type()
        self.model_name = self.config_loader.get_model_name()
        self.score_threshold = self.config_loader.get_score_threshold(self.model_type)
        self.use_count_hint = self.config_loader.get_use_count_hint(self.model_type)
        self.device = self._resolve_device()

        self.processor, self.model = self.load_model()

    def _resolve_device(self) -> torch.device:
        """
        Resolve the compute device from config.
        """
        cuda_devices = self.config_loader.get_cuda_devices()
        if torch.cuda.is_available() and cuda_devices:
            first_device = str(cuda_devices).split(",")[0].strip()
            return torch.device(f"cuda:{first_device}")
        return torch.device("cpu")

    def load_model(self) -> tuple[Sam3Processor, Sam3Model]:

        """
        Load Sam3Processor and Sam3Model from HuggingFace.
        Sam3Processor - handles image preprocessing and tokenisation. 
        Sam3Model - runs the segmentation forward pass.
        """

        model_params = self.config_loader.get_huggingface_load_kwargs_for_segmentation(self.model_type)

        logger.info("Loading SAM 3 model and processor - this may take a moment....")

        processor = Sam3Processor.from_pretrained(self.model_name)
        model = Sam3Model.from_pretrained(self.model_name, **model_params)
        model = model.to(self.device) # type: ignore
        model.eval()

        logger.info("SAM 3 model and processor loaded successfully.")

        return processor, model
    
    def generate_outputs(self, image: Image.Image, formatted_prompt: str):
        """
        Preprocess inputs and run the SAM 3 forward pass.

        This method does 3 things:
        1. Passes image + formatted text prompt through Sam3Processor to produce model-ready tensors
        2. Moves tensors to the correct device
        3. Runs Sam3Model forward pass under torch.no_grad()

        Returns the RAW model outputs - no post-processing applied.
        """

        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        logger.debug("Running SAM 3 forward pass...")

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs, inputs

        