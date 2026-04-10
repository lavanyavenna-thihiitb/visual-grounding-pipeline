import os

import logging
import torch
import numpy as np
from PIL import Image
from typing import Any, Optional

from transformers import Sam3Model, Sam3Processor

from visual_grounding.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class Sam3_Segmentation:

    def __init__(self, config_path: str, device: Optional[str] = None) -> None:
        self.config_loader = ConfigLoader(config_path)
        self.model_type = self.config_loader.get_model_type()
        self.model_name = self.config_loader.get_model_name()

        self.score_threshold = self.config_loader.get_score_threshold(self.model_type)
        self.use_count_hint = self.config_loader.get_use_count_hint(self.model_type)

        if device:
            self.device = torch.device(device)
        else:
            self.device = self._resolve_device()

        logger.info(f"The device ---- {self.device}")

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

        logger.info(f"The model is on the {model.device}")

        model.eval()

        logger.info("SAM 3 model and processor loaded successfully.")

        return processor, model
    
    def generate_masks_bboxes(self, image_path: str, formatted_prompt: str):
        """
        Run SAM3 on a single image path + prompt.

        Returns:
            inputs  : Processor outputs (model-ready tensors)
            outputs : Raw SAM3 model outputs
        """

        # Load image from path
        image = Image.open(image_path).convert("RGB")

        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = inputs.get("original_sizes")

        if target_sizes is None:
            raise ValueError("Missing 'original_sizes' in inputs for postprocessing")

        target_sizes = target_sizes.tolist()

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=0.5,
            target_sizes=target_sizes
        )[0]

        masks = results["masks"]
        bboxes = results["boxes"]
        scores = results["scores"]

        return {
            "masks": masks,
            "bboxes": bboxes,
            "scores": scores,
        }


if __name__ == "__main__":

    import logging
    from visual_grounding.utils.prompt_loader import format_prompt_for_entity_segmentation
    from visual_grounding.utils.visualize import visualize_masks

    logging.basicConfig(level=logging.INFO)

    image_path = "/fsxvision_new/pratyush.jena/Datasets/Indic-Laion/extracted_full_dataset/00365/003658320.jpg"

    caption_dict = {"caption_type": "short_caption", "caption": "A man with white hair and beard wearing sunglasses and a yellow vest over a light pink shirt, standing outdoors with trees in the background.", "entities": ["man with white hair and beard", "sunglasses", "yellow vest", "light pink shirt", "trees"], "counts": {"man with white hair and beard": 1, "sunglasses": 1, "yellow vest": 1, "light pink shirt": 1, "trees": -1}}

    PROMPT_PATH = "src/visual_grounding/prompts/entity_segmentation_prompt.yaml"

    formatted_prompt = format_prompt_for_entity_segmentation(caption_dict["entities"], caption_dict["counts"], PROMPT_PATH)

    logger.info("=== FORMATTED PROMPT ===")
    logger.info(formatted_prompt)

    # model
    CONFIG_PATH = "src/visual_grounding/config/entity_segmentation_config.yaml"
    segmentor = Sam3_Segmentation(CONFIG_PATH)

    #Results
    results = segmentor.generate_masks_bboxes(image_path, formatted_prompt)

    logger.info(f"The number of masks generated are: {len(results['masks'])}")

    logger.info("Results are generated!!!!")

    output_path = "output_visualization.jpg"

    visualize_masks(image_path=image_path, masks=results["masks"], output_path=output_path)

    logger.info(f"Saved visualization to {output_path}")
