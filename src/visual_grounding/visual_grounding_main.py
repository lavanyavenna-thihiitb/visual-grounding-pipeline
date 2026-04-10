import logging
import argparse
from pathlib import Path
import json
import shutil

from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.utils.load_jsonl import load_test_jsonl
from visual_grounding.entity_extraction_main import EntityExtractor

logger = logging.getLogger(__name__)


def load_from_test_jsonl(jsonl_path: Path) -> list[dict]:
    """
    Reads records from a test JSONL file.
 
    Expected format — one JSON object per line:
 
        {
            "image_path": "/path/to/image.jpg",
            "caption": {
                "short_caption":    "...",
                "medium_caption":   "...",
                "long_caption":     "...",
                "visual_caption":   "...",
                "semantic_caption": "..."
            }
        }
 
    Notes:
        - The key is "caption" (singular), not "captions".
        - All caption types are processed — short, medium, long,
          visual, and semantic.
        - Lines that are malformed or missing required keys are
          skipped with a warning.
    """
    records   = []
 
    if not jsonl_path.exists():
        raise FileNotFoundError(f"test_jsonl not found at: {jsonl_path}")
 
    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
 
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_no, e)
                continue
 
            if "image_path" not in record or "caption" not in record:
                logger.warning(
                    "Skipping line %d — missing 'image_path' or 'caption' key.",
                    line_no
                )
                continue
 
            records.append(record)
 
    logger.info("Loaded %d valid records from %s", len(records), jsonl_path)
    return records
 
def load_from_input_folder(input_folder: Path) -> list[dict]:
    """
    Reads image files from the input folder.
    Each image is treated as a record with no captions — placeholder
    for future non-test mode implementation.
    """
 
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found at: {input_folder}")
 
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
 
    for image_file in sorted(input_folder.rglob("*")):
        if image_file.suffix.lower() in image_extensions:
            records.append({
                "image_path": str(image_file),
                "caption": {},
            })
 
    logger.info(
        "Loaded %d images from input folder: %s", len(records), input_folder
    )
    return records
 

def build_flat_captions(records: list[dict], output_folder: Path) -> tuple[list[str], list[dict]]:
    """
    Flattens all captions from all records into a single list. 
    Builds a parallel index_map sor we can reconstruct per-image 
    results after extraction.

    Skips records whose output file already exist (resumability).

    Returns
        flat_captions: list[str]
            All captions to be processed, in order.
        index_map: list[dict]
            Parallel list - index_map[i] describes where flat_captions[i]
            came from (image_path, caption_type, catpiton_text)
    """
    flat_captions = []
    index_map = []
    skipped = 0

    for record in records:
        image_path = record["image_path"]
        img_stem = Path(image_path).stem
        output_file = output_folder / img_stem / "entities.jsonl"

        # If the output file already exists, this image has been full processed - skip it entirely
        if output_file.exists():
            logger.info("Skipping %s - already processed.", img_stem)
            skipped+=1
            continue

        captions = record.get("caption", {})
        for caption_type, caption_text in captions.items():
            if not caption_text or not caption_text.strip():
                logger.warning(
                    "Empty caption for image %s caption_type %s - skipping.", img_stem, caption_type 
                )

            flat_captions.append(caption_text)
            index_map.append({
                "image_path": image_path,
                "img_stem": img_stem,
                "caption_type": caption_type,
                "caption_text": caption_text,
            })

    logger.info("Built %d captions from %d records (%d skipped - already processed).", len(flat_captions), len(records) - skipped, skipped)

    return flat_captions, index_map


def write_results(index_map: list[dict], results: list[dict], output_folder: Path) -> None:
    """
    Reconstructs per-image results from the flat index_map + results lists and writes them to: output_folder / img_stem / entities.jsonl

    One line per caption type in the JSONL file.
    """
    per_image: dict[str, list[dict]] = {}

    for meta, result in zip(index_map, results):
        img_stem = meta["img_stem"]
        if img_stem not in per_image:
            per_image[img_stem] = []

        per_image[img_stem].append({
            "caption_type": meta["caption_type"],
            "caption": meta["caption_text"],
            "entities": result["entities"],
            "counts": result["counts"],
        })

    #Write one entities.jsonl per image
    for img_stem, lines in per_image.items():
        image_path = next( meta["image_path"] for meta in index_map if meta["img_stem"] == img_stem)
        img_output_dir = output_folder / img_stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = img_output_dir / "entities.jsonl"
        with open(output_file, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")


        logger.info("Written %d lines to %s", len(lines), output_file)

        # shutil.copy2 preserves metadata (timestamps etc.)
        src_image = Path(image_path)
        if src_image.exists():
            dst_image = img_output_dir / src_image.name
            shutil.copy2(src_image, dst_image)
            logger.info("Copied image to %s", dst_image)
        else:
            logger.warning(
                "Image not found at %s — skipping copy.", image_path
            )


def run_pipeline(extraction_config: str, test_mode: bool) -> None:
    """
    Orchestrates the full Stage 1 pipeline:
        1. Load config
        2. Load test JSONL
        3. Build flat captions + index map (with resumability)
        4. Run entity extraction in batches
        5. Write per-image results to disk
    """

    #Load config file

    logger.info("Loading config from %s", extraction_config)
    config = ConfigLoader(extraction_config)

    output_folder = Path(config.get_output_folder())
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Output folder: %s", output_folder)

    # Load test JSONL

    if test_mode:
        logger.info("Test mode — loading records from test JSONL.")
        records = load_from_test_jsonl(Path(config.get_test_jsonl()))

    else:
        logger.info("Production mode — loading records from input folder.")
        records = load_from_input_folder(Path(config.get_input_folder()))

    if not records:
        logger.error("No valid records found in test JSONL. Existing.")
        return
    
    # Build flat captions + index map
    flat_captions, index_map = build_flat_captions(records, output_folder)

    if not flat_captions:
        logger.info("All records already processed. Nothing to do.")
        return

    # Run entity extraction

    logger.info("Initialising EntityExtractor ...")
    extractor = EntityExtractor(extraction_config)
    logger.info("Running extraction on %d captions ...", len(flat_captions))
    results = extractor.extract(flat_captions)

    # Write results to disk
    write_results(index_map, results, output_folder)
    logger.info("Entity Extractin complete.")


def cli():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler("visual_grounding.log"),
            logging.StreamHandler(),
        ]
    )    

    parser = argparse.ArgumentParser(
        description="Visual Grounding Pipeline"
    )

    parser.add_argument(
        "--extraction_config",
        type=str,
        default="src/visual_grounding/config/entity_extraction_config.yaml",
        help="Path to entity_extraction_config.yaml",
    )

    parser.add_argument(
        "--test_extraction",
        action="store_true",
        default=False,
        help=("If set, reads records from the test JSONL file specified in config "
            "under paths.test_jsonl. If not set, reads images from paths.input_folder."),
    )

    args = parser.parse_args()

    run_pipeline(extraction_config=args.extraction_config,
                 test_mode=args.test_extraction)


if __name__ == "__main__":
    cli()