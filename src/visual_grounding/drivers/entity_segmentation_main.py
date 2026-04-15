import logging
import json
from pathlib import Path
from typing import Optional, List, Tuple
from collections import OrderedDict, defaultdict

import numpy as np

from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.utils.prompt_loader import format_prompt_for_entity_segmentation
from visual_grounding.models.sam3_batch_model import Sam3_Batch_Segmentation

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
                "count": vlm_output[caption_type].get("count", {})
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
        batch_size: int = 100,
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

    batch = []
    metadata_map = {}
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

                # Process each caption separately to preserve caption_type
                for caption in transformed_record["captions"]:
                    caption_type = caption["caption_type"]
                    entities = caption.get("entities", [])
                    counts = caption.get("count", {})

                    entity_prompts = format_entities_with_prompts(entities, counts, prompt_path)

                    for entity_prompt in entity_prompts:
                        batch_item = {
                            "image_path": image_path,
                            "caption_type": caption_type,
                            "prompt": entity_prompt["prompt"],
                            "entity": entity_prompt["entity"],
                            "count": entity_prompt["count"]
                        }

                        # Track: which batch index this item is at
                        key = (image_path, caption_type, entity_prompt["entity"])
                        metadata_map[key] = len(batch)

                        batch.append(batch_item)
                        processed+=1

                        if len(batch) >= batch_size:
                            yield batch, metadata_map
                            batch = []
                            metadata_map = {}

                # # Extract unique entities and counts
                # entities, counts = get_unique_entities_with_counts(transformed_record["captions"])

                # # Format prompts for each entity
                # entity_prompts = format_entities_with_prompts(entities, counts, prompt_path)
                
                # # Create one batch item per entity
                # for item in entity_prompts:
                #     batch.append({
                #         "image_path": transformed_record["image_path"],
                #         "prompt": item["prompt"],
                #         "entity": item["entity"], 
                #         "count": item["count"]
                #     })

                #     processed+=1

                #     # Yield batch when full
                #     if len(batch) >= batch_size:
                #         yield batch
                #         batch = []
                

            except Exception as e:
                if logger:
                    logger.warning(f"Error processing record: {e}")
                continue
    
    if batch:
        yield batch, metadata_map

def save_mask(mask: np.ndarray, mask_dir: Path, entity_name: str, instance_idx: int) -> str:
    """Save mask to disk, return path."""
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Convert torch tensor to numpy if needed
    if hasattr(mask, 'cpu'):  # torch tensor
        mask = mask.cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Sanitize entity name for filename
    safe_entity_name = entity_name.replace(" ", "_").replace("/", "_")
    mask_path = mask_dir / f"{safe_entity_name}_mask_{instance_idx}.npy"
    
    np.save(str(mask_path), mask)
    return str(mask_path)
 
 
def reconstruct_output_structure(
    results: List[dict],
    metadata_map: dict,
    batch_records: List[dict],
    mask_output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> list:
    """
    Reconstruct results into output format grouped by image_path -> caption_type -> entities.
    
    Returns:
        {
            "image_path": "...",
            "captions": {
                "short_caption": {
                    "entities": [
                        {
                            "entity": "flag",
                            "count": 1,
                            "instances": [
                                {
                                    "mask_path": "masks/flag_mask_0.npy",
                                    "bbox": [...],
                                    "score": 0.95
                                }
                            ]
                        }
                    ]
                },
                ...
            }
        }
    """
    
    # Group results by (image_path, caption_type)
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result_idx, result in enumerate(results):
        image_path = result["image_path"]
        caption_type = result["caption_type"]
        entity = result["entity"]
        count = result["count"]
        masks = result.get("masks", [])
        bboxes = result.get("bboxes", [])
        scores = result.get("scores", [])
        
        instances = []
        
        # Process each mask/bbox/score tuple
        for instance_idx, (mask, bbox, score) in enumerate(zip(masks, bboxes, scores)):
            mask_path = save_mask(mask, mask_output_dir, entity, instance_idx)
            
            instances.append({
                "mask_path": mask_path,
                "bbox": bbox.cpu().tolist() if hasattr(bbox, 'cpu') else (bbox.tolist() if isinstance(bbox, np.ndarray) else list(bbox)),
                "score": float(score)
            })
        
        # Add entity entry
        entity_entry = {
            "entity": entity,
            "count": count,
            "instances": instances
        }
        
        grouped[image_path][caption_type].append(entity_entry)
    
    # Convert to final output format
    output_records = []
    seen_images = set()
    
    for image_path, captions_dict in grouped.items():
        if image_path in seen_images:
            continue
        seen_images.add(image_path)
        
        output_records.append({
            "image_path": image_path,
            "captions": dict(captions_dict)
        })
    
    return output_records

def run_batch_segmentation(config_path: str, logger: logging.Logger):

    config_loader = ConfigLoader(config_path)

    input_jsonl_file = Path(config_loader.get_input_jsonl_file())
    output_jsonl_file = Path(config_loader.get_output_jsonl_file())
    prompt_path = config_loader.get_prompt_path()
    batch_size = config_loader.get_batch_size() or 100
    mask_output_dir = Path(config_loader.get_mask_folder())

    logger.info(f"Input JSONL: {input_jsonl_file.resolve()}")
    logger.info(f"Output JSONL: {output_jsonl_file.resolve()}")
    logger.info(f"Prompt template: {prompt_path}")

    import pdb; pdb.set_trace()

    processed = 0
    total_output_records = []

    sam_segmentor = Sam3_Batch_Segmentation(config_path, logger, device="cuda:3")

    for batch_num, (batch, metadata_map) in enumerate(process_jsonl_to_batches(input_file=input_jsonl_file, prompt_path=prompt_path, batch_size=batch_size,logger=logger), start=1):

        processed += len(batch)
        logger.info(f"Batch {batch_num}: {len(batch)} records | Total processed: {processed}")

        try:
            results = sam_segmentor.generate_masks_bboxes_batch(batch)

            # valid_mask_count = count_results_with_masks(results)

            # Reconstruct output structure
            batch_output = reconstruct_output_structure(
                results=results,
                metadata_map=metadata_map,
                batch_records=batch,
                mask_output_dir=mask_output_dir,
                logger=logger
            )

            total_output_records.extend(batch_output)

            logger.info(f"Results are: {results}")

        except Exception as e:
            logger.info(f"Can't run sam segmentor!!!, {e}")

    logger.info(f"Writing {len(total_output_records)} records to {output_jsonl_file}")

    with open(output_jsonl_file, 'w') as f:
        for record in total_output_records:
            f.write(json.dumps(record) + '\n')
    
    logger.info("Pipeline complete")


if __name__ == "__main__":

    # Setting up logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Pass config file path
    config_path = "src/visual_grounding/config/batch_inference_sam_config.yaml"

    try:
        run_batch_segmentation(config_path, logger)
    except Exception as e:
        logger.critical(f"Fatal error in pipeline: {e}", exc_info=True)
        raise
    finally:
        logger.info("Script execution finished")
