import os

import time
import torch
import logging
import torch.multiprocessing as mp
from PIL import Image
from typing import Any, Optional, List, Dict
from math import ceil

from visual_grounding.config.config_loader import ConfigLoader

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


# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU worker class (identical logic to your original, no changes needed)
# ─────────────────────────────────────────────────────────────────────────────

class Sam3_Batch_Segmentation:

    def __init__(self, config_path: str, logger: logging.Logger,
                 device: Optional[str] = None, global_counter: int = 1) -> None:
        self.config_loader   = ConfigLoader(config_path=config_path)
        self.model_type      = self.config_loader.get_model_type()
        self.model_name      = self.config_loader.get_model_name()
        self.global_counter  = global_counter
        self.score_threshold = self.config_loader.get_score_threshold(self.model_type)
        self.device          = torch.device(device) if device else self._resolve_device()
        self._setup_torch_runtime()
        self.logger = logger
        self.logger.info(f"Device being used to load SAM3 model is: {self.device}")
        self.model         = self.load_model()
        self.transform     = self._build_transform()
        self.postprocessor = self._build_postprocessor()

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
            first_device = str(cuda_devices).split(',')[0].strip()
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

    def _create_datapoint(self):
        return Datapoint(find_queries=[], images=[])

    def _set_image(self, datapoint, pil_image):
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])] # type: ignore

    def _add_text_prompt(self, datapoint, text_query):
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
        return self.global_counter - 1

    def load_model(self):
        from dependencies.sam3.sam3 import build_sam3_image_model
        bpe_path = self.config_loader.get_bpe_path()
        model = build_sam3_image_model(bpe_path=bpe_path)
        model.eval().to(self.device)
        self.logger.info("SAM3 native model loaded!")
        return model

    def generate_masks_bboxes_batch(self, inputs: List[Dict]) -> List[Dict]:
        datapoints    = []
        prompt_id_map = {}

        for item in inputs:
            image     = Image.open(item["image_path"]).convert("RGB")
            datapoint = self._create_datapoint()
            self._set_image(datapoint, image)

            for prompt_str, entity, count in zip(item["prompt"], item["entities"], item["count"]):
                pid = self._add_text_prompt(datapoint, prompt_str)
                prompt_id_map[pid] = {
                    "image_path": item["image_path"],
                    "entity":     entity,
                    "count":      count,
                    "prompt":     prompt_str,
                }

            datapoint = self.transform(datapoint)
            datapoints.append(datapoint)

        batch     = collate(datapoints, dict_key="dummy")["dummy"]
        batch     = copy_data_to_device(batch, self.device)

        with torch.inference_mode():
            outputs = self.model(batch)

        processed = self.postprocessor.process_results(outputs, batch.find_metadatas) # type: ignore

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
# Worker process: lives for the entire run, owns exactly ONE GPU
# ─────────────────────────────────────────────────────────────────────────────

def _worker_process(
    rank:         int,
    gpu_id:       int,
    config_path:  str,
    in_queue:     mp.Queue,
    out_queue:    mp.Queue,
):
    """
    Spawned once per GPU. Loops forever pulling (chunk_id, inputs) off
    in_queue, runs inference, and pushes (chunk_id, results) to out_queue.
    Exits cleanly when it receives None.
    """
    # Per-process logger (goes to stderr; no file contention)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s — %(message)s",
    )
    log = logging.getLogger(f"worker_gpu{gpu_id}")

    log.info(f"Worker starting on cuda:{gpu_id}")

    # Each worker gets its own model replica on its own GPU
    # global_counter is offset by rank * 10^6 so IDs never collide across GPUs
    segmentor = Sam3_Batch_Segmentation(
        config_path    = config_path,
        logger         = log,
        device         = f"cuda:{gpu_id}",
        global_counter = rank * 1_000_000,   # ← avoids prompt_id collisions
    )

    log.info("Model loaded, entering inference loop")

    while True:
        item = in_queue.get()

        # Poison pill → clean shutdown
        if item is None:
            log.info("Received shutdown signal, exiting")
            break

        chunk_id, inputs = item

        try:
            results = segmentor.generate_masks_bboxes_batch(inputs)
            out_queue.put((chunk_id, results))
        except Exception as e:
            log.error(f"Inference failed for chunk {chunk_id}: {e}", exc_info=True)
            out_queue.put((chunk_id, []))   # send empty so main doesn't hang


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: splits work, dispatches, collects, preserves order
# ─────────────────────────────────────────────────────────────────────────────

class MultiGPUSam3Orchestrator:
    """
    Keeps 8 (or N) persistent worker processes alive.
    Call .run(inputs) as many times as you want — workers stay warm.
    Call .shutdown() when completely done.
    """

    def __init__(self, config_path: str, gpu_ids: List[int]):
        # spawn is MANDATORY for CUDA — fork + CUDA = deadlock
        mp.set_start_method("spawn", force=True)

        self.gpu_ids     = gpu_ids
        self.num_workers = len(gpu_ids)

        # One input queue per worker (keeps routing deterministic)
        # maxsize=2 gives gentle backpressure; tune to taste
        self.in_queues  = [mp.Queue(maxsize=2) for _ in gpu_ids]
        self.out_queue  = mp.Queue()            # single shared output queue

        self.workers: List[mp.Process] = []
        for rank, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target  = _worker_process,
                args    = (rank, gpu_id, config_path,
                           self.in_queues[rank], self.out_queue),
                daemon  = True,     # dies automatically if main process crashes
            )
            p.start()
            self.workers.append(p)

        print(f"[Orchestrator] Spawned {self.num_workers} workers on GPUs {gpu_ids}")

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, inputs: List[Dict]) -> List[Dict]:
        """
        Splits `inputs` evenly across all GPUs, dispatches, waits for all
        results, and returns them in the ORIGINAL input order.
        """
        chunks    = self._split(inputs, self.num_workers)
        num_chunks = len(chunks)

        # Round-robin dispatch
        for chunk_id, chunk in enumerate(chunks):
            worker_idx = chunk_id % self.num_workers
            self.in_queues[worker_idx].put((chunk_id, chunk))

        # Collect — order comes back scrambled, sort by chunk_id
        received: Dict[int, List[Dict]] = {}
        for _ in range(num_chunks):
            chunk_id, results = self.out_queue.get()
            received[chunk_id] = results

        # Reassemble in original order
        ordered_results = []
        for chunk_id in range(num_chunks):
            ordered_results.extend(received[chunk_id])

        return ordered_results

    def shutdown(self):
        """Send poison pill to every worker and wait for clean exit."""
        for q in self.in_queues:
            q.put(None)
        for p in self.workers:
            p.join()
        print("[Orchestrator] All workers shut down cleanly")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split(lst: List, n: int) -> List[List]:
        """Split list into n roughly-equal chunks."""
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

    # ── build your input list ─────────────────────────────────────────────────
    single_input = {
        "image_path": "/fsxvision_new/pratyush.jena/Datasets/Indic-Laion/"
                      "extracted_full_dataset/02087/020874131.jpg",
        "prompt":   ["glasses"],
        "entities": ["glasses"],
        "count":    [1],
    }
    total_inputs = [single_input] * 110      # 110 items, just like your original

    # ── spin up orchestrator once ─────────────────────────────────────────────
    orchestrator = MultiGPUSam3Orchestrator(
        config_path = config_path,
        gpu_ids     = [1, 2, 3, 4, 5, 6, 7],   # all 8 GPUs
    )

    # ── run inference (call as many times as you want) ────────────────────────
    t0      = time.time()
    results = orchestrator.run(total_inputs)
    elapsed = time.time() - t0

    # with open("temp.json", "w") as file:
    #     json.dump()
    # ── clean shutdown ────────────────────────────────────────────────────────
    orchestrator.shutdown()

    import pdb; pdb.set_trace()
    
    log.info(f"Total results: {len(results)}  |  Time: {elapsed:.2f}s")
    log.info("Done!")

