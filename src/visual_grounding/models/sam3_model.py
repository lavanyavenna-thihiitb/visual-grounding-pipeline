import os

import logging
import torch
import numpy as np
from PIL import Image
from typing import Any

from transformers import Sam3Model, Sam3Processor

from visual_grounding.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class Sam3Segementation:

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

        Single-image inference. Thin wrapper around the batch method for backward compatibility.

        Returns the RAW model outputs - no post-processing applied.
        """

        outputs, inputs = self.generate_batch_outputs(
            images=[image],
            formatted_prompts=[[formatted_prompt]],
        )

        return outputs, inputs
    
    def generate_batch_outputs(self, images: list[Image.Image], formatted_prompts: list[list[str]],):
        """
        Batched inference across multiple images and their respective entity prompts.

        Args:
            images: List of PIL images, one per sample.
            formatted_prompts: List of entity prompt lists, one list per image. e.g. [["car","person"],["tree","dog"],["wheel"]] each inner list can have a different
            number of entites - the processor pads them to the longest list in the batch.

        returns:
            outputs: Raw SAM 3 model outputs for the full batch.
            inputs: The processor-generated tensors that were fed to the model.

        Notes:
            - Images are encoded together in a single vision backbone forward pass.
            - Text prompts are padded to the maximum entity count in the batch using
              padding=True, so variable-length prompt lists are handled automatically.
            - outputs.pred_masks shape: [batch_size, num_queries, H, W]
              Slice by batch index to get masks for a specific image.
        """

        if len(images)!=len(formatted_prompts):
            raise ValueError(
                f"Mismatch: {len(images)} images but {len(formatted_prompts)} prompt lists."
            )
        
        logger.debug(
            "Running SAM 3 batched forward pass - "
            f"{len(images)} images, prompt counts: {[len(p) for p in formatted_prompts]}"
        )

        inputs = self.processor(
            images=images,
            text=formatted_prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs, inputs
    

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    CONFIG_PATH = "src/visual_grounding/config/entity_segmentation_config.yaml"
    DATASET_PATH = Path("test_dataset.jsonl")
    OUTPUTS_ROOT = Path("outputs")
    BATCH_SIZE = 4

    logger.info("Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        dataset = [
            json.loads(line)
            for line in f
            if line.strip() # skip empty lines
        ]

    records = []    # List of {"image_path", "stem", "caption_type", "entities", "counts"}

    for entry in dataset:
        image_path = Path(entry["image_path"])
        stem = image_path.stem
        jsonl_path = OUTPUTS_ROOT / stem / "entities.jsonl"

        if not image_path.exists():
            logger.warning(f"Image not found, skipping: {image_path}")
            continue

        if not jsonl_path.exists():
            logger.warning(f"entities.jsonl not found, skipping: {jsonl_path}")
            continue

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                records.append({
                    "image_path": image_path,
                    "stem":       stem,
                    "caption_type": row["caption_type"],
                    "entities": row["entities"],
                    "counts": row["counts"],
                })

    logger.info(f"Built {len(records)} records (image x capton_type pairs).")

    if not records:
        logger.error("No valid records to process. Exiting.")
        sys.exit(1)


    # Initialise model
    logger.info("Initialising Sam3Segmentation model ...")
    segmentor = Sam3Segementation(CONFIG_PATH)

    def format_entity_prompts(
        entities: list[str],
        counts: dict[str, int],
        use_count_hint: bool,
    ) -> list[str]:
        if use_count_hint:
            return [
                f"{entity} (count: {counts.get(entity, 1)})"
                for entity in entities
            ]
        return list(entities)

    # Batch inference
    all_results = []

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch_records = records[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(
            f"Processing batch {batch_num}/{total_batches}"
            f"({len(batch_records)})"
        ) 

        images: list[Image.Image] = [
            Image.open(rec["image_path"]).convert("RGB")
            for rec in batch_records
        ]

        formatted_prompts: list[list[str]] = [
            format_entity_prompts(
                rec["entities"],
                rec["counts"],
                segmentor.use_count_hint,
            )
            for rec in batch_records
        ]

        logger.info(
            "Batch prompt summary: "
            + ", ".join(
                f"{rec['stem']}({rec['caption_type']})={prompts}"
                for rec, prompts in zip(batch_records, formatted_prompts)
            )
        )

        try:
            outputs, inputs = segmentor.generate_batch_outputs(
                images=images,
                formatted_prompts=formatted_prompts,
            )

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}", exc_info=True)
            continue

        for i, rec in enumerate(batch_records):
            mask = outputs.pred_masks[i]
            logger.info(
                f"  [{rec['stem']} | {rec['caption_type']}] "
                f"pred_masks shape: {tuple(mask.shape)} "
                f"| entities: {rec['entities']}"
            )

        all_results.append({
            "records": batch_records,
            "outputs": outputs,
            "inputs":  inputs,
        })
        
    logger.info(
        f"Inference complete. Processed"
        f"{sum(len(b['records']) for b in all_results)} records across "
        f"{len(all_results)} batches."
    )