from pathlib import Path
import logging
import json
from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.drivers.multi_threading import MultiModel
from visual_grounding.models.sam3_model import Sam3_Segmentation
from visual_grounding.utils.prompt_loader import format_prompt_for_entity_segmentation

logger = logging.getLogger(__name__)

def find_image_file(img_folder: Path) -> Path | None:
    """
    Finds the first valid image file in the folder.
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for file in img_folder.iterdir():
        if file.suffix.lower() in img_extensions:
            return file
    return None

def iter_valid_folders(input_folder: Path):
    """
    Generator that yields valid img_folders one at a time.

    Validation rules the following structure:
       input_folder/
            img_folder_1/
                entities.jsonl <- REQUIRED
                image.jpg      <- REQUIRED
                
            img_folder_2/
                entities.jsonl <- REQUIRED
                image.jpg      <- REQUIRED 
    """

    # If input_folder path does not exist
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder.resolve()}")
    
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder path is not a directory: {input_folder.resolve()}") 
    
    # Flag to check if img_folder exists or not
    found_any = False
    
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

        yield img_folder

    if not found_any:
        raise ValueError(f"No subdirectories found in {input_folder.resolve()}.")



# def validate_input_folder(input_folder: Path) -> list[Path]:
#     """
#     Validates the structure of the input folder and returns a list of valid image subdirectories.

#     Expected structure:
#         input_folder/
#             img_folder_1/
#                 entities.jsonl <- REQUIRED
#                 image.jpg      <- REQUIRED
                
#             img_folder_2/
#                 entities.jsonl <- REQUIRED
#                 image.jpg      <- REQUIRED

#     validation rules:
#         1. input_folder must exist and be a directory
#         2. input_folder must contain at least one subdirectory
#         3. Each subdirectory must contain both entities.jsonl and image.jpg
#             - If either is missing, the folder is skipped with a warning

#     Parameters:
#         input_folder: Path = Path to the root input folder. 

#     Returns:
#         list[Path] - List of valid img_folder Paths - those that have both required files.
#     """

#     # If input_folder path does not exist
#     if not input_folder.exists():
#         raise FileNotFoundError(f"Input folder does not exist: {input_folder.resolve()}")
    
#     # If input_folder is not a directory
#     if not input_folder.is_dir():
#         raise NotADirectoryError(f"Input folder path is not a directory: {input_folder.resolve()}")
    
#     # List of all subdirectories present in the input_folder
#     all_subdirs = [p for p in input_folder.iterdir() if p.is_dir()]

#     if not all_subdirs:
#         raise ValueError(f"Input folder {input_folder.resolve()} does not contain any subdirectories.")
    
#     logger.info("Found %d subdirectories in input folder: %s", len(all_subdirs), input_folder.resolve())

#     # Each subdirectory must contain both required files
#     valid_folders: list[Path] = []
#     skipped = 0

#     for img_folder in sorted(all_subdirs):

#         entities_jsonl = img_folder / "entities.jsonl"
#         image_jpg = find_image_file(img_folder) 

#         if not entities_jsonl.exists() or image_jpg is None:
#             logger.warning(f"Skipping {img_folder.name} - missing required files(s) either entities.jsonl or image.jpg")

#             skipped += 1
#             continue

#         valid_folders.append(img_folder)

#     logger.info(f"Input folder validation complete - {len(valid_folders)} valid / {skipped} skipped out of {len(all_subdirs)} total.")


#     if not valid_folders:
#         raise ValueError(f"No valid img_folders found in {input_folder.resolve()}.")
    
#     return valid_folders


# def get_unprocessed_captions(valid_folders: list[Path]) -> list[dict]:
#     """
#     For each valid img_folder, reads entities.jsonl and checks which caption types have already been processed by looking for a 
#     {caption_type}_results.json file in the same folder. 

#     Only caption types whose results file is absent are included in the outputs. 
#     Images where every caption type is already processed are skipped entirely.

#     Parameters
#         valid_folders: list[Path]

#     Returns
#         list[WorkItem]
#         One WorkItem per image that still has at least one unprocessed caption type. Each WorkItem has the shape::
#         {
#             "image_path" : str,
#             "image_folder" : Path,
#             "captions" : [
#             {
#                 "caption_type": "short_caption",
#                 "caption": "...",
#                 "entities": [...],
#                 "counts" : {...},
#             }

#             ....
            
#             ]
        
#         }
#     """

#     work_items: list[dict] = []

#     fully_processed = 0
#     partially_processed = 0
#     parse_errors = 0

#     for img_folder in valid_folders:

#         entities_jsonl = img_folder / "entities.jsonl"
#         image_path = str((img_folder / "image"))  ### Shiiittttt we made a mistake here - the image has a name not image.jpg


def unprocessed_captions_for_image(img_folder: Path) -> dict | None:

    """
    Reads entities.jsonl for a single image folder and constructs a work item.

    Returns:
        dict | None
    """

    entities_file = img_folder / "entities.jsonl"
    image_file = find_image_file(img_folder)

    captions = []

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

                caption_type = data.get("caption_type")
                caption = data.get("caption")
                entities = data.get("entities",[])
                counts = data.get("counts", {})

                if not caption_type or not caption:
                    logger.warning(f"Missing caption type or caption in {entities_file}")
                    continue

                captions.append({
                    "caption_type": caption_type,
                    "caption": caption,
                    "entities": entities,
                    "counts": counts,
                })

    except Exception as e:
        logger.error(f"Failed reading {entities_file}: {e}")
        return None
    
    if not captions:
        logger.warning(f"No valid captions found in {img_folder.name}")
        return None
    
    return {
        "image_path": str(image_file),
        "image_folder": img_folder,
        "captions": captions
    }

def get_image_entities(work_item: dict) -> dict:
    """
    Combines entities across all captions into a unique set, builds counts, and generates a single prompt.

    Returns:
        dicts with:
            image_path
            image_folder
            entities
            counts
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

    return {
        "image_path": work_item["image_path"],
        "image_folder": work_item["image_folder"],
        "entities": unique_entities,
        "counts": all_counts
    }

def build_multimodel_inputs(img_entities: dict, prompt_path: str) -> list[dict]:
    """
    Creates one input per entity with explicit mapping.

    Returns:
        List of inputs for Multimodel
    """

    image_path = img_entities["image_path"]
    entities = img_entities["entities"]
    counts = img_entities["counts"]

    inputs = []

    for entity in entities:
        count = counts.get(entity, 1)

        #Build prompt for single entity
        prompt = format_prompt_for_entity_segmentation(
            entites=[entity],
            counts={entity: count},
            prompt_file=prompt_path
        )

        inputs.append({
            "image": image_path,
            "prompt": prompt,
            "entity": entity,
            "count": count
        })

    return inputs


def run_segmentation(config_path):

    config_loader = ConfigLoader(config_path)

    input_folder = Path(config_loader.get_input_folder())
    output_folder = Path(config_loader.get_output_folder())
    prompt_path = config_loader.get_prompt_path()

    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")

    processed = 0

    for img_folder in iter_valid_folders(input_folder):

        work_item = unprocessed_captions_for_image(img_folder)

        if not work_item:
            logger.warning(f"No captions to be processed in {img_folder.name}")
            continue

        # Merge all entities → run ONCE → reuse results
        img_entities = get_image_entities(work_item)

        # Convert the image entities into prompts
        inputs = build_multimodel_inputs(img_entities, prompt_path)

        # Model initialization
        multimodel = MultiModel(
            model_cls=Sam3_Segmentation,
            config_path=config_path,
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # or [0] if single GPU
        )

        # pass the inputs to run the model
        outputs = multimodel.predict(inputs)

        print(outputs)

        break

    # Validate input folder structure and get list of valid img_folders 
    # valid_folders = validate_input_folder(input_folder)

    # logger.info(f"Proceeding with {len(valid_folders)} valid image folder(s).")

    # # Filter out already-processed caption types per image
    # work_items = get_unprocessed_captions(valid_folders)

    # if not work_items:
    #     logger.info("Nothing to process - all caption types for all images have results. Exiting.")
    #     return
    
    # logger.info(f"{len(work_items)} image(s) with pending caption types queued for segmentation.")

    return None


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    #Change config path if needed
    config_path = "src/visual_grounding/config/entity_segmentation_config.yaml"

    #Run segmentation pipeline
    run_segmentation(config_path)