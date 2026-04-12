from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class ModelSegmentation:
    """
    Generic wrapper for a model on a specific GPU

    It does 3 things - 
        1. Instantiate the model on a specific GPU
        2. Store how to run inference (infer_fn)
        3. Process a batch sequentially on that GPU

    However, with the current SAM3 model that we are using, it only deals with one input at a time. There is no batch processing. 
    So, it is this class is going to process one input at a time instead of a batch.
    """

    def __init__(self, model_cls: Callable, config_path: str, gpu_id: int) -> None:
        
        self.gpu = gpu_id

        # Set device
        torch.cuda.set_device(gpu_id)
        # Initialize model
        self.model = model_cls(config_path, device=f"cuda:{gpu_id}")

        logger.info(f"Segmenation model initialized on GPU {gpu_id}")

    def __call__(self, batch: List[Any]) -> List[Any]:

        outputs = []
        
        #Processes a batch sequentially (no internal batching) - SAM3
        for item in batch:
            outputs.append(self.model.generate_masks_bboxes(item, item["image"], item["prompt"]))

        return outputs
    
    def unload(self):
        logger.info(f"Deleting Segmentation model on the GPU {self.gpu}")
        del self.model


class MultiModel:
    """
    Takes a list of inputs -> distributes them across GPUs -> runs in parallel -> returns results in original order

                MultiModel (Manager)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    Replica 0       Replica 1       Replica 2
    (GPU 0)         (GPU 1)         (GPU 2)
        │               │               │
    chunk A         chunk B         chunk C
        │               │               │
        └─────── merge results back ────

    """

    def __init__(self, model_cls: Callable, config_path: str, gpu_ids: List[int]) -> None:
        
        if not gpu_ids:
            raise ValueError("GPU ids are empty")

        logger.info(f"Initializing MultiModel on GPUs: {gpu_ids}")

        self.segments = [ModelSegmentation(model_cls, config_path, gid) for gid in gpu_ids]

    def predict(self, inputs: List[Any]) -> List[Any]:

        if len(self.segments) == 1:
            return self.segments[0](inputs)
        
        # Round robin split
        split_inputs = [[] for _ in self.segments] #[] here is the bucket

        for idx, item in enumerate(inputs):
            ridx = idx % len(self.segments)
            split_inputs[ridx].append(item)

        # logging
        for i, chunk in enumerate(split_inputs):
            if chunk:
                logger.info(f"Segmentation {i} (GPU {self.segments[i].gpu}) processing {len(chunk)} samples")

        # Parallel execution
        merged = [None] * len(inputs)

        with ThreadPoolExecutor(max_workers=len(self.segments)) as pool:
            futs = {}

            for k, seg in enumerate(self.segments):
                # If there is no bucket(split_inputs) for kth GPU/ segment
                if not split_inputs[k]:
                    continue

                futs[pool.submit(seg, split_inputs[k])] = k

            for fut in as_completed(futs):
                k = futs[fut]
                sub_out = fut.result()

                for loc_i, result in enumerate(sub_out):
                    global_idx = k + loc_i * len(self.segments)
                    merged[global_idx] = result

        return merged
    
    def unload(self):
        for s in self.segments:
            s.unload()



