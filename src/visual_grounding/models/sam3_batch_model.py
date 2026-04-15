import os
import time
import torch
import logging
from PIL import Image
from typing import Any, Optional, List, Dict

from visual_grounding.config.config_loader import ConfigLoader

# SAM3 native imports
import dependencies.sam3
from dependencies.sam3.sam3.train.data.collator import collate_fn_api as collate
from dependencies.sam3.sam3.model.utils.misc import copy_data_to_device
from dependencies.sam3.sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint
)

from dependencies.sam3.sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

from dependencies.sam3.sam3.eval.postprocessors import PostProcessImage

class Sam3_Batch_Segmentation:

    def __init__(self, config_path: str, logger: logging.Logger, device: Optional[str] = None, global_counter:int = 1) -> None:
        self.config_loader = ConfigLoader(config_path=config_path)
        self.model_type = self.config_loader.get_model_type()
        self.model_name = self.config_loader.get_model_name()
        self.global_counter = global_counter

        self.score_threshold = self.config_loader.get_score_threshold(self.model_type)

        self.device = torch.device(device) if device else self._resolve_device()

        self._setup_torch_runtime()

        self.logger = logger
        self.logger.info(f"Device being used to load SAM3 model is: {self.device}")

        # Load the model
        self.model = self.load_model()
        self.transform = self._build_transform()
        self.postprocessor = self._build_postprocessor()

    def _setup_torch_runtime(self):
        # Set CUDA device explicitly
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set autocast dtype
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # Inference mode. Disable if you need gradients
        torch.inference_mode().__enter__()


    def _resolve_device(self) -> torch.device:
        cuda_devices = self.config_loader.get_cuda_devices()
        if torch.cuda.is_available() and cuda_devices:
            first_device = str(cuda_devices).split(',')[0].strip()
            return torch.device(f"cuda:{first_device}")        
        return torch.device("cpu")
    
    def _build_transform(self):
        transform_params = self.config_loader.get_sam3_batch_transform_params(self.model_type)

        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(**transform_params["random_resize_params"]),
                ToTensorAPI(),
                NormalizeAPI(**transform_params["normalize_params"])
            ]
        )
        return transform
    
    def _build_postprocessor(self):
        postprocessor_params = self.config_loader.get_sam3_batch_postprocess_params(self.model_type)
        postprocessor = PostProcessImage(**postprocessor_params)
        return postprocessor
    
    def _create_datapoint(self):
        return Datapoint(find_queries=[], images=[])

    def _set_image(self, datapoint, pil_image):
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])] #type: ignore

    def _add_text_prompt(self, datapoint, text_query):
        global GLOBAL_COUNTER

        w, h = datapoint.images[0].size

        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=self.global_counter, 
                    original_image_id=self.global_counter,
                    original_category_id=1,
                    original_size=[w, h], #type: ignore
                    object_id=0,
                    frame_index=0,
                )
            )
        )

        self.global_counter += 1
    
    def load_model(self):
        from dependencies.sam3.sam3 import build_sam3_image_model

        bpe_path = self.config_loader.get_bpe_path()

        model = build_sam3_image_model(bpe_path=bpe_path)
        model.eval().to(self.device)

        self.logger.info(f"SAM3 native model loaded!")

        return model
    
    def generate_masks_bboxes_batch(self, inputs: List[Dict]) -> List[Dict]:

        datapoints = []
        raw_images = []

        # Build datapoints
        for item in inputs:

            image = Image.open(item["image_path"]).convert("RGB")

            datapoint = self._create_datapoint()
            self._set_image(datapoint, image)
            self._add_text_prompt(datapoint, item["prompt"]) # However, we may have multiple prompts for a single image, so we gotta deal with that 

            datapoint = self.transform(datapoint)

            datapoints.append(datapoint)
            raw_images.append(image)

        # Collate -> batch
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device)

        # Forward 
        with torch.inference_mode():
            # import pdb; pdb.set_trace()
            outputs = self.model(batch)

        # Postprocessor
        processed = self.postprocessor.process_results(
            outputs,
            batch.find_metadatas #type: ignore
        )

        # print(f"The total time to execute was - {time.time() - a}")

        # 5. Format results (match your old structure)
        results = []
        for i in range(1, len(processed)+1):

            input_item = inputs[i - 1]  # align with processed index

            results.append({
                "image_path": input_item["image_path"],
                "caption_type": input_item["caption_type"],
                "entity": input_item["entity"],
                "count": input_item["count"],
                "prompt": input_item["prompt"],
                "masks": processed[i]["masks"],
                "bboxes": processed[i]["boxes"],
                "scores": processed[i]["scores"],
            })

        return results
    

if __name__ == "__main__":

    # Log
    logging.basicConfig(filename="logs/sam3_batch.log",
                        filemode="a",
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    
    log = logging.getLogger(__name__)
    
    # Model
    config_path = "src/visual_grounding/config/batch_inference_sam_config.yaml"
    sam_segmentor = Sam3_Batch_Segmentation(config_path, log, device="cuda:3")

    # Inputs
    total_inputs = []
    input = {}

    input["image_path"] = "/fsxvision_new/pratyush.jena/Datasets/Indic-Laion/extracted_full_dataset/02087/020874131.jpg"
    input["prompt"] = "glasses"
    input["entity"] = "glasses"
    input["count"] = {"glasses": 1}

    total_inputs.append(input)

    #results
    results = sam_segmentor.generate_masks_bboxes_batch(inputs=total_inputs*110)

    log.info(f"The number of masks generated are: {len(results)}")
    log.info("Results are generated!!!!!!")

    


