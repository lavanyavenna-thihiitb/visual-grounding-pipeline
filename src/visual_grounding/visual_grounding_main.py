import logging
import argparse
from pathlib import Path
import json

from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.utils.load_jsonl import load_test_jsonl
from visual_grounding.entity_extraction_main import EntityExtractor

logger = logging.getLogger(__name__)

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

        captions = record.get("captions", {})
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
            img_output_dir = output_folder / img_stem
            img_output_dir.mkdir(parents=True, exist_ok=True)

            output_file = img_output_dir / "entities.jsonl"
            with open(output_file, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")


            logger.info("Written %d lines to %s", len(lines), output_file)


def run_pipeline(extraction_config: str) -> None:
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

    records = load_test_jsonl(config.get_test_jsonl())

    if not records:
        logger.error("No valid records found in test JSONL. Existing.")
        return
    
    # Build flat captions + index map
    flat_captions, index_map = build_flat_captions(records, output_folder)

    if not flat_captions:
        logger.info("All records already processed. Nothing to do.")

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
        required=True,
        help="Path to entity_extraction_config.yaml",
    )

    args = parser.parse_args()

    run_pipeline(extraction_config=args.extraction_config)


if __name__ == "__main__":
    cli()