# Also use plot_results method to save visualizations to the folder
# Add multiprocessor

import logging
import json
from pathlib import Path
from typing import Optional, List, Tuple
from collections import OrderedDict, defaultdict

import numpy as np

from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.utils.prompt_loader import format_prompt_for_entity_segmentation
from visual_grounding.models.sam3_batch_cache_model import Sam3_Batch_Segmentation
from visual_grounding.utils.visualize import visualize_masks

def load_processed_set(processed_file: Path) -> set:
    """Load already-processed image paths into a set for O(1) lookup."""
    if not processed_file.exists():
        return set()
    with open(processed_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}
    
def mark_as_processed(processed_file: Path, image_path: str):
    """Append a single image path to the processed log."""
    with open(processed_file, "a") as f:
        f.write(image_path + "\n")

def transform_record(record: dict) -> dict:
    """
    Transform single jsonl record from input format to restructured format.
    
    Input:
        {
            "path": "...",
            "prompt_str": "...",
            "vlm_output": "{...json...}"
        }
    
    Output:
        {
            "image_path": "...",
            "captions": [
                {"caption_type": "short_caption", "entities": [...], "counts": {...}},
                ...
            ]
        }
    """

    vlm_output = json.loads(record["vlm_output"])

    captions = []
    for caption_type in ["short_caption", "medium_caption", "long_caption", "visual_caption", "semantic_caption"]:

        if caption_type in vlm_output:
            captions.append({
                "caption_type": caption_type,
                "entities": vlm_output[caption_type].get("entities", []),
                "count": vlm_output[caption_type].get("counts", {})
            })

    return {
        "image_path": record["path"],
        "captions": captions
    }
    
def get_unique_entities_with_counts(captions: List[dict]) -> Tuple[List[str], dict]:
    """
    Extract unique entities across all captions and their counts (first occurrence).
    
    Args:
        captions: list of caption dicts with "entities" and "counts" keys
    
    Returns:
        (unique_entities_list, merged_counts_dict)
    """

    unique_entities = OrderedDict() # Preserved order of first occurence

    for caption in captions:
        entities = caption.get("entities", [])
        counts = caption.get("count", {})

        for entity in entities:
            if entity not in unique_entities:
                unique_entities[entity] = counts.get(entity, 1)

    return list(unique_entities.keys()), dict(unique_entities)


# Helper function
def count_results_with_masks(results: List[dict]) -> int:
    """
    Count how many results contain non-empty masks.

    Args:
        results: list of result dicts from SAM3

    Returns:
        int: number of results with at least one mask
    """
    count = 0

    for item in results:
        masks = item.get("masks")

        # Handle different possible formats safely
        if masks is not None and len(masks) > 0:
            count += 1

    return count


def format_entities_with_prompts(entities: List[str], counts: dict[str, int], prompt_path: str) -> List[dict]:

    """
    Generate individual prompts per entity.

    Args:
        entities: list of entity strings
        counts: dict mapping entity -> count
        prompt_file: path to YAML config

    Returns:
        List of dicts:
        [
            {
                "entity": str,
                "count": int,
                "prompt": str
            },
            ...
        ]
    """

    prompts = []

    for entity in entities:
        count = counts.get(entity, 1)

        # Calling formating function
        prompt = format_prompt_for_entity_segmentation(entites=[entity], counts={entity: count}, prompt_file=prompt_path)

        prompts.append(({
            "entity": entity,
            "count": count,
            "prompt": prompt
        }))

    return prompts



def process_jsonl_to_batches(
        input_file: Path,
        prompt_path: str, 
        processed_file_path: Path,
        batch_size: int = 3,
        logger: Optional[logging.Logger] = None):
    
    """
    Stream process jsonl file, transform records, extract entities, format prompts.
    Yield batches of size batch_size.
    
    Args:
        input_file: path to input.jsonl
        prompt_path: path to prompt YAML template
        batch_size: number of records per batch (default 100)
        logger: logging instance
    
    Yields:
        (batch_records, metadata_map)
        list of batch records, each containing:
        {
            "image_path": str,
            "prompt": str (formatted multi-entity prompt),
            "entities": list[str],
            "counts": dict[str, int]
        }
        - metadata_map: dict to reconstruct output by caption_type
    """

    already_processed = load_processed_set(processed_file_path)  # loaded once, O(1) lookups

    if logger:
        logger.info(f"Resuming: {len(already_processed)} images already processed")

    batch = []
    metadata = {}
    processed = 0

    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # Transform record
                transformed_record = transform_record(record)
                image_path = transformed_record["image_path"]
                captions = transformed_record["captions"]

                if image_path in already_processed:
                    continue

                if image_path not in metadata:
                    metadata[image_path] = {
                        "image_path": image_path,
                        "captions": captions
                    }

                if logger:
                    logger.info(f"Fetching unique entities and their counts for image - {image_path}")

                # Extract unique entities and counts
                entities, counts = get_unique_entities_with_counts(transformed_record["captions"])

                if logger:
                    logger.info(f"Formatting prompts for the entities and counts......")

                # Format prompts for each entity
                entity_prompts = format_entities_with_prompts(entities, counts, prompt_path)

                if logger:
                    logger.info(f"The total number of unique entities for image - {image_path} are {len(entity_prompts)}")

                batch.append({
                    "image_path": transformed_record["image_path"],
                    "prompt": [ep["prompt"] for ep in entity_prompts],
                    "entities": [ep["entity"] for ep in entity_prompts],
                    "count": [ep["count"] for ep in entity_prompts]
                })

                processed+=1

                if len(batch) >= batch_size:
                    
                    if logger:
                        logger.info(f"Yielding batch of size: {len(batch)} along with it's metadata")

                    yield batch, metadata
                    batch = []
                    metadata = {}
                

            except Exception as e:
                if logger:
                    logger.warning(f"Error processing record: {e}")
                continue
    
    if batch:
        if logger:
            logger.info(f"Yielding batch of size: {len(batch)} along with it's metadata")

        yield batch, metadata

def save_mask(mask: np.ndarray, mask_dir: Path, entity_name: str, img_path: str, instance_idx: int) -> str:
    """Save mask to disk, return path."""
    # mask_dir.mkdir(parents=True, exist_ok=True)

    image_id = Path(img_path).stem
    subfolder = mask_dir / image_id
    subfolder.mkdir(parents=True, exist_ok=True)

    # Convert torch tensor to numpy if needed
    if hasattr(mask, 'cpu'):  # torch tensor
        mask = mask.cpu().numpy() #type: ignore
    elif not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Sanitize entity name for filename
    safe_entity_name = entity_name.replace(" ", "_").replace("/", "_")
    mask_path = subfolder / f"{safe_entity_name}_mask_{instance_idx}.npy"
    
    np.save(str(mask_path), mask)
    return str(mask_path)

def group_results_by_image_and_entity(results, mask_output_dir, logger):

    grouped = {}

    logger.info(f"Grouping results by image and entity.....")

    for result in results:
        image_path = result["image_path"]
        entity = result["entity"]
        count = result["count"]

        masks = result.get("masks", [])
        bboxes = result.get("bboxes", [])
        scores = result.get("scores", [])

        # Build instances
        instances = []

        for idx, (mask, bbox, score) in enumerate(zip(masks, bboxes, scores)):
            mask_path = save_mask(mask, mask_output_dir, entity, image_path, idx)

            instances.append({
                "mask_path": mask_path,
                "bbox": bbox.cpu().tolist() if hasattr(bbox, 'cpu') else (bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)),
                "score": float(score)
            })

        entity_entry = {
            "entity": entity,
            "count": count,
            "instances": instances
        }

        # Initialize nested dicts
        if image_path not in grouped:
            grouped[image_path] = {}

        grouped[image_path][entity] = entity_entry

    return grouped


def save_outputs(metadata: dict, grouped_results_by_img_entities: dict, output_folder: Path, logger: Optional[logging.Logger] = None):
    """
        For each image and each caption_type, write a {caption_type}_entities.json
        under output_folder / <image_stem> / {caption_type}_entities.json.

        Args:
        metadata: {
            image_path: {
                "image_path": str,
                "captions": [
                    {"caption_type": str, "entities": [str, ...], "count": {str: int}},
                    ...
                ]
            }
        }
        grouped_results_by_img_entities: {
            image_path: {
                entity_name: {
                    "entity": str,
                    "count": int,
                    "instances": [{"mask_path": str, "bbox": [...], "score": float}, ...]
                }
            }
        }
        output_folder: root folder to write per-image per-caption JSON files
    """

    for image_path, meta in metadata.items():
        image_stem = Path(image_path).stem
        image_output_dir = output_folder / image_stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Get the segmentation results for this image (may be empty if SAM found nothing)
        entity_results = grouped_results_by_img_entities.get(image_path, {})

        for caption in meta["captions"]:
            caption_type = caption["caption_type"]
            caption_entities = caption.get("entities", [])
            caption_counts = caption.get("count", {})

            entities_output = []

            for entity in caption_entities:
                if entity in entity_results:
                    entities_output.append(entity_results[entity])
                else:
                    entities_output.append({
                        "entity": entity,
                        "count": caption_counts.get(entity, 1),
                        "instances": []
                    })

            output = {
                "image": Path(image_path).name,
                "caption_type": caption_type,
                "entities": entities_output
            }

            out_file = image_output_dir / f"{caption_type}_entities.json"
            with open(out_file, "w") as f:
                json.dump(output, f, indent=4)

            if logger:
                logger.info(f"Saved {out_file} for image {image_path}")

        # --- Visualization (once per image, across all unique entities) ---
        # if entity_results:
        #     visualize_masks(
        #         image_path=image_path,
        #         entity_results=entity_results,
        #         output_dir=image_output_dir,
        #         logger=logger
        #     )
        # else:
        #     if logger:
        #         logger.info(f"No segmentation results for {image_stem}, skipping visualization")
 
 
def run_batch_segmentation(config_path: str, logger: logging.Logger):

    config_loader = ConfigLoader(config_path)

    input_jsonl_file = Path(config_loader.get_input_jsonl_file())
    mask_output_dir = Path(config_loader.get_mask_folder())
    output_folder = Path(config_loader.get_output_folder())
    processed_file = Path(config_loader.get_processed_file())
    prompt_path = config_loader.get_prompt_path()
    batch_size = config_loader.get_batch_size() or 3
    cuda_device = f"cuda:{config_loader.get_cuda_devices()}"

    logger.info(f"Loaded the configurations...")
    logger.info(f"Input JSONL: {input_jsonl_file.resolve()}")
    logger.info(f"Output directory for storing masks is: {mask_output_dir.resolve()}")
    logger.info(f"Output directory for storing ouputs is: {output_folder.resolve()}")
    logger.info(f"File that stores the images processed so far is: {processed_file.resolve()}")
    logger.info(f"Prompt template: {prompt_path}")

    processed = 0

    sam_segmentor = Sam3_Batch_Segmentation(config_path, logger, device=cuda_device)

    for batch_num, (batch, metadata) in enumerate(process_jsonl_to_batches(input_file=input_jsonl_file, prompt_path=prompt_path, processed_file_path=processed_file, batch_size=batch_size,logger=logger), start=1):

        processed += len(batch)
        logger.info(f"Batch {batch_num}: {len(batch)} records | Total processed: {processed}")

        try:

            results = sam_segmentor.generate_masks_bboxes_batch(batch)

            logger.info(f"Generated segmentation masks for - {len(results)}")

            valid_mask_count = count_results_with_masks(results)

            logger.info(f"Number of valid mask count is: {valid_mask_count} out of {len(results)}")

            grouped_results_by_img_entities = group_results_by_image_and_entity(results, mask_output_dir, logger)

            # Reconstruct output structure
            save_outputs(
                metadata=metadata,
                grouped_results_by_img_entities=grouped_results_by_img_entities,
                output_folder=output_folder,
                logger=logger
            )

            logger.info(f"Saved ouputs for batch of images")

            # Mark each image in this batch as done
            for item in batch:
                mark_as_processed(processed_file, item["image_path"])

            logger.info(f"Number of processed images so far are: {processed}")

        except Exception as e:
            logger.info(f"Can't run sam segmentor!!!, {e}")

    
    logger.info("Pipeline complete")


if __name__ == "__main__":

    # Setting up logger
    logging.basicConfig(
        filename="/opt/dlami/nvme/lavanya.venna/visual_grounding_outputs/logs/segmentation_batch_7.log",
        filemode="a",
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
        force=True
    )

    logger = logging.getLogger(__name__)

    # Pass config file path
    config_path = "src/visual_grounding/config/batch_inference_sam_config.yaml"

    logger.info(f"Loaded the config file from {config_path} to run batch segmentation using SAM3")

    try:
        run_batch_segmentation(config_path, logger)
    except Exception as e:
        logger.critical(f"Fatal error in pipeline: {e}", exc_info=True)
        raise
    finally:
        logger.info("Script execution finished")
