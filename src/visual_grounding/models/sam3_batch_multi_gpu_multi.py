import os

import time
import torch
import logging
import torch.multiprocessing as mp
from math import ceil
from PIL import Image
from typing import Optional, List, Dict
from collections import Counter

from visual_grounding.config.config_loader import ConfigLoader

import dependencies.sam3
from dependencies.sam3.sam3.train.data.collator import collate_fn_api as collate
from dependencies.sam3.sam3.model.utils.misc import copy_data_to_device
from dependencies.sam3.sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from dependencies.sam3.sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from dependencies.sam3.sam3.eval.postprocessors import PostProcessImage


# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU model class
# ─────────────────────────────────────────────────────────────────────────────

class Sam3_Batch_Segmentation:

    def __init__(
        self,
        config_path: str,
        logger: logging.Logger,
        device: Optional[str] = None,
    ) -> None:
        self.config_loader   = ConfigLoader(config_path=config_path)
        self.model_type      = self.config_loader.get_model_type()
        self.model_name      = self.config_loader.get_model_name()
        self.score_threshold = self.config_loader.get_score_threshold(self.model_type)
        self.device          = torch.device(device) if device else self._resolve_device()

        self._setup_torch_runtime()

        self.logger = logger
        self.logger.info(f"Device: {self.device}")

        self.model         = self._load_model()
        self.transform     = self._build_transform()
        self.postprocessor = self._build_postprocessor()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _setup_torch_runtime(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()

    def _resolve_device(self) -> torch.device:
        cuda_devices = self.config_loader.get_cuda_devices()
        if torch.cuda.is_available() and cuda_devices:
            first_device = str(cuda_devices).split(",")[0].strip()
            return torch.device(f"cuda:{first_device}")
        return torch.device("cpu")

    def _build_transform(self):
        p = self.config_loader.get_sam3_batch_transform_params(self.model_type)
        return ComposeAPI(transforms=[
            RandomResizeAPI(**p["random_resize_params"]),
            ToTensorAPI(),
            NormalizeAPI(**p["normalize_params"]),
        ])

    def _build_postprocessor(self):
        p = self.config_loader.get_sam3_batch_postprocess_params(self.model_type)
        return PostProcessImage(**p)

    def _load_model(self):
        from dependencies.sam3.sam3 import build_sam3_image_model
        bpe_path = self.config_loader.get_bpe_path()
        model    = build_sam3_image_model(bpe_path=bpe_path)
        model.eval().to(self.device)
        self.logger.info("SAM3 model loaded")
        return model

    # ── datapoint helpers ─────────────────────────────────────────────────────

    def _create_datapoint(self):
        return Datapoint(find_queries=[], images=[])

    def _set_image(self, datapoint, pil_image):
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]  # type: ignore

    def _add_text_prompt(self, datapoint, text_query: str, prompt_id: int):
        """prompt_id is a local integer assigned by the caller — no instance state."""
        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=prompt_id,
                    original_image_id=prompt_id,
                    original_category_id=1,
                    original_size=[w, h],  # type: ignore
                    object_id=0,
                    frame_index=0,
                ),
            )
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def generate_masks_bboxes_batch(self, inputs: List[Dict]) -> List[Dict]:
        """
        inputs: list of dicts, each with keys:
            image_path : str
            prompt     : List[str]
            entities   : List[str]
            count      : List[int]
        """
        datapoints    = []
        prompt_id_map = {}
        local_counter = 0           # fresh every call, no shared state

        for item in inputs:
            image     = Image.open(item["image_path"]).convert("RGB")
            datapoint = self._create_datapoint()
            self._set_image(datapoint, image)

            for prompt_str, entity, count in zip(
                item["prompt"], item["entities"], item["count"]
            ):
                pid = local_counter
                local_counter += 1
                self._add_text_prompt(datapoint, prompt_str, pid)
                prompt_id_map[pid] = {
                    "image_path": item["image_path"],
                    "entity":     entity,
                    "count":      count,
                    "prompt":     prompt_str,
                }

            datapoint = self.transform(datapoint)
            datapoints.append(datapoint)

        # Collate → batch → device
        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device)

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(batch)

        # Postprocess
        processed = self.postprocessor.process_results(
            outputs, batch.find_metadatas  # type: ignore
        )

        # Build results list
        results = []
        for pid, meta in prompt_id_map.items():
            r = processed.get(pid, {})
            results.append({
                "image_path": meta["image_path"],
                "entity":     meta["entity"],
                "count":      meta["count"],
                "prompt":     meta["prompt"],
                "masks":      r.get("masks",  []),
                "bboxes":     r.get("boxes",  []),
                "scores":     r.get("scores", []),
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Worker process — one per entry in gpu_ids, owns exactly one model replica
# ─────────────────────────────────────────────────────────────────────────────

def _worker_process(
    rank:        int,       # unique worker index across all workers
    gpu_id:      int,       # physical GPU (can repeat for multi-replica per GPU)
    config_path: str,
    in_queue:    mp.Queue,
    out_queue:   mp.Queue,
):
    logging.basicConfig(
        level  = logging.INFO,
        format = f"%(asctime)s [Worker {rank} | GPU {gpu_id}] %(levelname)s — %(message)s",
    )
    log = logging.getLogger(f"worker_{rank}_gpu{gpu_id}")
    log.info("Starting up")

    segmentor = Sam3_Batch_Segmentation(
        config_path = config_path,
        logger      = log,
        device      = f"cuda:{gpu_id}",
    )

    log.info("Model ready — entering inference loop")

    while True:
        item = in_queue.get()

        if item is None:            # poison pill → clean shutdown
            log.info("Shutting down")
            break

        chunk_id, inputs = item
        try:
            results = segmentor.generate_masks_bboxes_batch(inputs)
            out_queue.put((chunk_id, results))
        except Exception as e:
            log.error(f"Chunk {chunk_id} failed: {e}", exc_info=True)
            out_queue.put((chunk_id, []))   # never leave main process hanging


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiGPUSam3Orchestrator:
    """
    Spin up persistent workers defined entirely by the gpu_ids list.

    Examples
    --------
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]         →  8 workers, 1 per GPU
    gpu_ids = [0, 0, 1, 1, 2, 2, 3, 3]         →  8 workers, 2 per GPU
    gpu_ids = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7] → 11 workers, 4 on GPU 0
    """

    def __init__(self, config_path: str, gpu_ids: List[int]):
        mp.set_start_method("spawn", force=True)   # mandatory for CUDA

        self.gpu_ids     = gpu_ids
        self.num_workers = len(gpu_ids)

        self.in_queues = [mp.Queue(maxsize=2) for _ in range(self.num_workers)]
        self.out_queue = mp.Queue()

        self.workers: List[mp.Process] = []
        for rank, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target = _worker_process,
                args   = (rank, gpu_id, config_path,
                          self.in_queues[rank], self.out_queue),
                daemon = True,
            )
            p.start()
            self.workers.append(p)

        summary = ", ".join(
            f"GPU {g} × {n}" for g, n in sorted(Counter(gpu_ids).items())
        )
        print(f"[Orchestrator] {self.num_workers} workers — {summary}")

    def run(self, inputs: List[Dict]) -> List[Dict]:
        """
        Split inputs across all workers, wait for results,
        return in original input order.
        """
        chunks     = self._split(inputs, self.num_workers)
        num_chunks = len(chunks)

        # Round-robin dispatch
        for chunk_id, chunk in enumerate(chunks):
            self.in_queues[chunk_id % self.num_workers].put((chunk_id, chunk))

        # Collect (arrives out of order)
        received: Dict[int, List[Dict]] = {}
        for _ in range(num_chunks):
            chunk_id, results = self.out_queue.get()
            received[chunk_id] = results

        # Reassemble in original order
        ordered = []
        for chunk_id in range(num_chunks):
            ordered.extend(received[chunk_id])
        return ordered

    def shutdown(self):
        for q in self.in_queues:
            q.put(None)
        for p in self.workers:
            p.join()
        print("[Orchestrator] All workers shut down cleanly")

    @staticmethod
    def _split(lst: List, n: int) -> List[List]:
        if not lst:
            return []
        chunk_size = ceil(len(lst) / n)
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    log = logging.getLogger(__name__)

    config_path = "src/visual_grounding/config/batch_inference_sam_config.yaml"

    # ── build inputs ──────────────────────────────────────────────────────────
    single_input = {
        "image_path": "/fsxvision_new/pratyush.jena/Datasets/Indic-Laion/"
                      "extracted_full_dataset/02087/020874131.jpg",
        "prompt":   ["glasses"],
        "entities": ["glasses"],
        "count":    [1],
    }
    total_inputs = [single_input] * 110

    # ── configure workers via gpu_ids only ────────────────────────────────────
    #
    #   1 model per GPU  →  [0, 1, 2, 3, 4, 5, 6, 7]
    #   2 models per GPU →  [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    #   mixed            →  [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    #
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]   # ← change this line only

    orchestrator = MultiGPUSam3Orchestrator(
        config_path = config_path,
        gpu_ids     = gpu_ids,
    )

    # ── run ───────────────────────────────────────────────────────────────────
    t0      = time.time()
    results = orchestrator.run(total_inputs)
    elapsed = time.time() - t0

    orchestrator.shutdown()

    import pdb; pdb.set_trace()

    log.info(f"Total results : {len(results)}")
    log.info(f"Time elapsed  : {elapsed:.2f}s")
    log.info("Done!")
