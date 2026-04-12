"""
Entity Segmentation Pipeline
 
This module orchestrates the entity segmentation workflow:
1. Validates input folder structure - there is a certain input folder that is expected for this code to work. 
   The structure is mentioned below. 
2. Reads entity captions from JSONL files
3. Generates segmentation prompts for each entity
4. Runs multi-GPU inference using SAM3
5. Processes and saves masks with metadata
6. Visualizes results
 
Expected folder structure:
    input_folder/
        img_folder_1/
            entities.jsonl
            image.jpg
        img_folder_2/
            entities.jsonl
            image.jpg
"""

from pathlib import Path
import logging
import json
import numpy as np
import shutil
from typing import Any, Dict, List, Optional, Iterator
from visual_grounding.utils.visualize import visualize_masks_for_output_directory
from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.drivers.multi_threading import MultiModel
from visual_grounding.models.sam3_model import Sam3_Segmentation
from visual_grounding.utils.prompt_loader import format_prompt_for_entity_segmentation

def find_image_file(img_folder: Path) -> Optional[Path]:
    """
    Finds the first valid image file in the given folder.
    
    Supported formats: .jpg, .jpeg, .png, .webp
    
    Args:
        img_folder (Path): Directory to search for image files
        
    Returns:
        Optional[Path]: Path to first valid image file, or None if not found
        
    Raises:
        None (gracefully returns None if no image found)
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for file in img_folder.iterdir():
        if file.suffix.lower() in img_extensions:
            return file
    return None

def safe_name(name: str) -> str:
    """
    Converts entity names to filesystem-safe format.
    
    Replaces spaces with underscores to prevent filesystem issues.
    
    Args:
        name (str): Original entity name
        
    Returns:
        str: Filesystem-safe name
    """
    return name.replace(" ", "_")

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def iter_valid_folders(input_folder: Path, logger: logging.Logger) -> Iterator[Path]:
    """
    Generator that yields valid image folders one at a time.
    
    Validates folder structure:
        - input_folder must exist and be a directory
        - Each subdirectory must contain:
            * entities.jsonl (REQUIRED)
            * image file: .jpg, .jpeg, .png, or .webp (REQUIRED)
    
    Args:
        input_folder (Path): Root directory containing image subdirectories
        
    Yields:
        Path: Valid image folder paths
        
    Raises:
        FileNotFoundError: If input_folder does not exist
        NotADirectoryError: If input_folder is not a directory
        ValueError: If no valid subdirectories found
    """

    # If input_folder path does not exist
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder.resolve()}")
    
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder path is not a directory: {input_folder.resolve()}") 
    
    # Flag to check if img_folder exists or not
    found_any = False
    
    # Iterate through subdirectories and validate structure
    for img_folder in input_folder.iterdir():

        if not img_folder.is_dir():
            continue

        entities_jsonl = img_folder / "entities.jsonl"
        image_file = find_image_file(img_folder)

        if not entities_jsonl.exists() or image_file is None:
            logger.warning(
                f"Skipping {img_folder.name} - missing entities.jsonl or image file"
            )
            continue

        # Found valid folder
        found_any = True
        logger.debug(f"Validating folder: {img_folder.name}")
        yield img_folder

    if not found_any:
        raise ValueError(f"No subdirectories found in {input_folder.resolve()}.")


def unprocessed_captions_for_image(img_folder: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:

    """
    Reads entities.jsonl for a single image folder and constructs a work item.
    
    Expected JSONL format (one JSON object per line):
    {
        "caption_type": "str",
        "caption": "str",
        "entities": ["entity1", "entity2"],
        "counts": {"entity1": 2, "entity2": 1}
    }
    
    Args:
        img_folder (Path): Image folder containing entities.jsonl
        
    Returns:
        Optional[Dict[str, Any]]: Work item dict with keys:
            - image_path (str): Absolute path to image file
            - image_folder (Path): Image folder path
            - captions (List[Dict]): List of caption objects
            Returns None if no valid captions found or on error
            
    Raises:
        None (logs errors and returns None on failure)
    """

    entities_file = img_folder / "entities.jsonl"
    image_file = find_image_file(img_folder)

    if image_file is None:
        logger.error(f"No image file found in {img_folder}")
        return None

    captions: List[Dict[str, Any]] = []
    logger.info(f"Reading captions from {entities_file}")

    try:
        with open(entities_file, "r") as f:
            for line_num, line in enumerate(f, start=1):

                line = line.strip()

                #If line is empty
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"JSON parse error in {entities_file} at line {line_num}")
                    continue

                # Extract caption fields
                caption_type = data.get("caption_type")
                caption = data.get("caption")
                entities = data.get("entities",[])
                counts = data.get("counts", {})

                # Validate required fields
                if not caption_type or not caption:
                    logger.warning(
                        f"Missing caption_type or caption in {entities_file} at line {line_num}"
                    )
                    continue

                captions.append({
                    "caption_type": caption_type,
                    "caption": caption,
                    "entities": entities,
                    "counts": counts,
                })

                logger.debug(f"Loaded caption: {caption_type} with {len(entities)} entities")

    except Exception as e:
        logger.error(f"Failed reading {entities_file}: {e}")
        return None
    
    if not captions:
        logger.warning(f"No valid captions found in {img_folder.name}")
        return None
    
    logger.info(f"Successfully loaded {len(captions)} captions from {img_folder.name}")
    
    return {
        "image_path": str(image_file),
        "image_folder": img_folder,
        "captions": captions
    }

def get_image_entities(work_item: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Merges entities across all captions into a unified set with count resolution.
    
    When the same entity appears in multiple captions with different counts,
    the maximum count is used (assumes higher count is more accurate).
    
    Args:
        work_item (Dict[str, Any]): Work item from unprocessed_captions_for_image()
        
    Returns:
        Dict[str, Any]: Merged entity dict with keys:
            - image_path (str): Path to image
            - image_folder (Path): Image folder path
            - entities (List[str]): Unique entity names
            - counts (Dict[str, int]): Entity -> count mapping
    """

    all_counts: dict[str, int] = {}

    # Merge entities accross captions
    for caption in work_item["captions"]:
        entities = caption.get("entities", [])
        counts = caption.get("counts", {})

        for entity in entities:
            count = counts.get(entity, 1)

            if entity not in all_counts:
                all_counts[entity] = count
            else:
                # If entity exists in all_counts but with a different count number, choose the max count
                all_counts[entity] = max(all_counts[entity], count)

    # Unique entities 
    unique_entities = list(all_counts.keys())

    logger.info(
        f"Merged to {len(unique_entities)} unique entities: {', '.join(unique_entities)}"
    )

    return {
        "image_path": work_item["image_path"],
        "image_folder": work_item["image_folder"],
        "entities": unique_entities,
        "counts": all_counts
    }


# ============================================================================
# MODEL INPUT PREPARATION
# ============================================================================
 
def build_multimodel_inputs(img_entities: Dict[str, Any], prompt_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Creates model input dictionaries, one per entity.
    
    Each input contains:
    - Image and folder paths
    - Entity-specific prompt
    - Entity name and count metadata
    
    Args:
        img_entities (Dict[str, Any]): Output from get_image_entities()
        prompt_path (str): Path to prompt template file
        
    Returns:
        List[Dict[str, Any]]: List of input dicts for MultiModel.predict()
        
    Raises:
        FileNotFoundError: If prompt_path does not exist (raised by format_prompt_for_entity_segmentation)
    """

    image_path = img_entities["image_path"]
    entities = img_entities["entities"]
    counts = img_entities["counts"]
    image_folder = img_entities["image_folder"]

    inputs: List[Dict[str, Any]] = []
    logger.info(f"Building model inputs for {len(entities)} entities")

    for entity in entities:
        count = counts.get(entity, 1)

        #Build prompt for single entity
        prompt = format_prompt_for_entity_segmentation(
            entites=[entity],
            counts={entity: count},
            prompt_file=prompt_path
        )

        inputs.append({
            "sub_folder": image_folder,
            "image": image_path,
            "prompt": prompt,
            "entity": entity,
            "count": count
        })
        logger.debug(f"Created input for entity '{entity}' (count={count})")

    logger.info(f"Generated {len(inputs)} model inputs")
    return inputs

# ============================================================================
# OUTPUT PROCESSING
# ============================================================================

def process_outputs(outputs: List[Dict[str, Any]], img_folder: Path, work_item: Dict[str, Any], output_folder: Path, logger: logging.Logger) -> Path:

    """
    Processes model outputs and organizes masks, metadata, and results.
    
    Output structure:
        output_folder/
            img_folder_name/
                image.jpg (copied)
                masks/
                    entity_safe_mask_0.npy
                    entity_safe_mask_1.npy
                    ...
                caption_type_entities.json (one per caption type)
    
    Args:
        outputs (List[Dict[str, Any]]): Model predictions from MultiModel.predict()
        img_folder (Path): Original image folder (for naming)
        work_item (Dict[str, Any]): Original work item data
        output_folder (Path): Root output directory
        
    Returns:
        Path: Path to the image output directory
    """

    image_name = Path(work_item["image_path"]).name
    logger.info(f"Processing outputs for image: {image_name}")

    # Create output structure
    image_output_dir = output_folder / img_folder.name
    masks_dir = image_output_dir / "masks"

    image_output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directories: {image_output_dir}")

    source_image_path = Path(work_item["image_path"])
    destination_image_path = image_output_dir / image_name
    shutil.copy2(source_image_path, destination_image_path)
    logger.debug(f"Copied image to: {destination_image_path}")

    # Group outpust by entity
    entity_map: Dict[str, Dict[str, Any]] = {}
    logger.debug(f"Processing {len(outputs)} model outputs")

    for output in outputs:
        entity = output["entity"]
        count = output.get("count", 1)

        masks = output["masks"].cpu().numpy()
        bboxes = output["bboxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        instances: List[Dict[str, Any]] = []
        logger.debug(f"Processing entity '{entity}' with {len(masks)} instances")

        # Save each mask and create instance metadata
        for i in range(len(masks)):
            entity_safe = safe_name(entity)
            mask_filename = f"{entity_safe}_mask_{i}.npy"
            mask_path = masks_dir / mask_filename

            # Save mask as numpy array
            np.save(mask_path, masks[i])
            logger.debug(f"Saved mask: {mask_filename}")

            instances.append({
                "mask_path": str(Path("masks") / mask_filename),
                "bbox": bboxes[i].tolist(),
                "score": float(scores[i])
            })

        entity_map[entity] = {
            "entity": entity,
            "count": count,
            "instances": instances
        }

        logger.info(
            f"Entity '{entity}': saved {len(instances)} masks, "
            f"scores: {[f'{s:.3f}' for s in scores]}"
        )

    # Create JSON per caption_type
    for caption in work_item["captions"]:
        caption_type = caption["caption_type"]
        caption_entities = caption["entities"]

        result = {
            "image": image_name,
            "caption_type": caption_type,
            "entities": []
        }

        for ent in caption_entities:
            if ent in entity_map:
                result["entities"].append(entity_map[ent])

        json_path = image_output_dir / f"{caption_type}_entities.json"

        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        logger.info(f"Wrote caption metadata: {json_path}")

    logger.info(f"Output processing complete: {image_output_dir}")
    return image_output_dir


def run_segmentation(config_path: str, logger: logging.Logger) -> None:

    """
    Executes the complete entity segmentation pipeline.
    
    Pipeline steps:
    1. Load configuration from YAML
    2. Initialize multi-GPU SAM3 model
    3. For each valid image folder:
        a. Load captions and entities
        b. Build model inputs
        c. Run segmentation inference
        d. Process and save outputs
        e. Visualize results
    
    Args:
        config_path (str): Path to entity_segmentation_config.yaml
        logger_obj (logging.Logger): Logger instance
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If no valid image folders found
    """

    logger.info("=" * 80)
    logger.info("Starting Entity Segmentation Pipeline")
    logger.info("=" * 80)

    # Load configuration
    config_loader = ConfigLoader(config_path)
    input_folder = Path(config_loader.get_input_folder())
    output_folder = Path(config_loader.get_output_folder())
    prompt_path = config_loader.get_prompt_path()

    logger.info(f"Input folder: {input_folder.resolve()}")
    logger.info(f"Output folder: {output_folder.resolve()}")
    logger.info(f"Prompt template: {prompt_path}")

    processed = 0

    # Initialize multi-GPU model
    logger.info("Initializing SAM3 model on 8 GPUs")
    multimodel = MultiModel(
        model_cls=Sam3_Segmentation,
        config_path=config_path,
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # or [0] if single GPU
    ) 

    for img_folder in iter_valid_folders(input_folder, logger):

        logger.info(f"Processing image folder: {img_folder.name}")

        work_item = unprocessed_captions_for_image(img_folder, logger)

        if not work_item:
            logger.warning(f"No captions to be processed in {img_folder.name}")
            continue

        # Merge all entities → run ONCE → reuse results
        img_entities = get_image_entities(work_item, logger)
        logger.info(f"Total unique entities: {len(img_entities['entities'])}")

        # Convert the image entities into prompts
        inputs = build_multimodel_inputs(img_entities, prompt_path, logger)
        logger.info(f"Running inference on {len(inputs)} entities")

        # Run segmentation
        try:
            outputs = multimodel.predict(inputs)
            logger.info(f"Inference complete, received {len(outputs)} outputs")
        except Exception as e:
            logger.error(f"Inference failed for {img_folder.name}: {e}")
            continue

        # Process and save outputs
        try:
            img_output_folder = process_outputs(outputs, img_folder, work_item, output_folder, logger)
            logger.info(f"Outputs saved to: {img_output_folder}")
        except Exception as e:
            logger.error(f"Output processing failed for {img_folder.name}: {e}")
            continue

        # Visualize results
        try:
            logger.info("Generating visualizations")
            visualize_masks_for_output_directory(img_output_folder, logger)
            logger.info("Visualizations complete")
        except Exception as e:
            logger.error(f"Visualization failed for {img_folder.name}: {e}")

        processed += 1
        logger.info(f"Successfully processed: {img_folder.name}")

        break

     # Cleanup
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline complete. Processed {processed} image folder(s)")
    logger.info("Unloading models and freeing GPU memory")
    logger.info(f"{'=' * 80}\n")
    
    multimodel.unload()
    logger.info("All models unloaded successfully")


if __name__ == "__main__":

    logging.basicConfig(
        filename="entity_segmenation_log.log",
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Entity Segmentation Script Started")
    logger.info("=" * 80)

    #Change config path if needed
    config_path = "src/visual_grounding/config/entity_segmentation_config.yaml"

    try:
        # Run segmentation pipeline
        run_segmentation(config_path, logger)
    except Exception as e:
        logger.critical(f"Fatal error in pipeline: {e}", exc_info=True)
        raise
    finally:
        logger.info("Script execution finished")