import logging
from pathlib import Path
import json
from typing import Iterator, Dict, Any, Optional, List

from visual_grounding.config.config_loader import ConfigLoader

def parse_vlm_output(vlm_output_str: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Parse VLM output JSON string.
 
    Args:
        vlm_output_str (str): JSON string from vlm_output field
        logger (logging.Logger): Logger instance
 
    Returns:
        Optional[Dict[str, Any]]: Parsed VLM output or None if error
 
    Raises:
        None (gracefully returns None on error)
    """
    try:
        # VLM output is a JSON string, need to parse it
        if isinstance(vlm_output_str, str):
            parsed = json.loads(vlm_output_str)
        else:
            parsed = vlm_output_str
 
        return parsed
 
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse VLM output JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing VLM output: {e}")
        return None
    
def extract_captions_from_vlm(vlm_data: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Extract caption records from VLM output.
 
    VLM output format:
    {
        "short_caption": {
            "entities": ["entity1", "entity2"],
            "counts": {"entity1": 1, "entity2": 2}
        },
        "medium_caption": {...},
        ...
    }
 
    Converts to:
    [
        {
            "caption_type": "short_caption",
            "entities": ["entity1", "entity2"],
            "counts": {"entity1": 1, "entity2": 2}
        },
        ...
    ]
 
    Args:
        vlm_data (Dict[str, Any]): Parsed VLM output
        logger (logging.Logger): Logger instance
 
    Returns:
        List[Dict[str, Any]]: List of caption records
    """
    captions = []
 
    for caption_type, caption_data in vlm_data.items():
        if not isinstance(caption_data, dict):
            logger.warning(f"Skipping {caption_type}: not a dict")
            continue
 
        entities = caption_data.get("entities", [])
        counts = caption_data.get("counts", {})
 
        # Validate required fields
        if not entities:
            logger.debug(f"Skipping {caption_type}: no entities found")
            continue
 
        caption_record = {
            "caption_type": caption_type,
            "entities": entities,
            "counts": counts
        }
 
        captions.append(caption_record)
        logger.debug(f"Extracted caption type '{caption_type}' with {len(entities)} entities")
 
    return captions

def convert_record(raw_record: Dict[str, Any], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Convert a single raw VLM record to structured format.
 
    Args:
        raw_record (Dict[str, Any]): Raw record from input JSONL
        logger (logging.Logger): Logger instance
 
    Returns:
        Optional[Dict[str, Any]]: Structured record or None if conversion failed
    """

    # Extract image path
    image_path = raw_record.get("path")
    if not image_path:
        logger.warning(f"Record for {image_path} missing 'vlm_output' field")
        return None
    
    # Parse VLM output
    vlm_output_str = raw_record.get("vlm_output")
    if not vlm_output_str:
        logger.warning(f"Record for {image_path} missing 'vlm_output' field")
        return None
    
    vlm_data = parse_vlm_output(vlm_output_str, logger)
    if not vlm_data:
        logger.warning(f"Failed to parse VLM output for {image_path}")
        return None
    
    # Extract captions from VLM output
    captions = extract_captions_from_vlm(vlm_data, logger)
    if not captions:
        logger.warning(f"No valid captions extracted from {image_path}")
        return None

    structured_record = {
        "image_path": image_path,
        "captions": captions
    }

    logger.info(f"Converted record: {image_path} with {len(captions)} captions")
    return structured_record

def iter_raw_records(input_jsonl: str, logger: logging.Logger) -> Iterator[tuple[Dict[str, Any], int]]:
    """
    Generator that yields raw records from input JSONL file.
 
    Args:
        input_jsonl (str): Path to input JSONL file
        logger (logging.Logger): Logger instance
 
    Yields:
        tuple[Dict[str, Any], int]: (record dict, line number)
 
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    path = Path(input_jsonl)
 
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {path.resolve()}")
 
    logger.info(f"Reading input JSONL: {path.resolve()}")
 
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
 
            # Skip empty lines
            if not line:
                continue
 
            try:
                record = json.loads(line)
                yield record, line_num
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error at line {line_num}: {e}")
                continue

def convert_vlm_to_structure(input_jsonl_path: str, output_jsonl_path: str, logger: logging.Logger) -> int:
    """
    Convert raw VLM output JSONL to structured format JSONL.

    Args:
        input_jsonl (str): Path to input JSONL with VLM outputs
        output_jsonl (str): Path to output structured JSONL
        logger (logging.Logger): Logger instance

    Returns:
        int: Number of successfully converted records

    Raises:
        FileNotFoundError: If input file doesn't exist
    """

    converted = 0
    failed = 0

    output_path = Path(output_jsonl_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as out_f:
            for raw_record, line_num in iter_raw_records(input_jsonl_path, logger):

                logger.debug(f"Processing line: {line_num}")

                # Convert record
                structured_record = convert_record(raw_record, logger)

                if structured_record:
                    # Write to output
                    out_f.write(json.dumps(structured_record)+'\n')
                    converted+=1

                else:
                    failed += 1

    except Exception as e:
        logger.critical(f"Error during conversion: {e}", exc_info=True)
        raise

    logger.info("=" * 80)
    logger.info(f"Conversion complete!")
    logger.info(f"Successfully converted: {converted} records")
    logger.info(f"Failed: {failed} records")
    logger.info(f"Output saved to: {output_path.resolve()}")
    logger.info("=" * 80)
 
    return converted



def main(config_path: str) -> None:
    """
        Main function to run the conversion.

        Args:
            config_path: path that contains the configuration for segmentation
    """
    config_loader = ConfigLoader(config_path)

    logging.basicConfig(
        filename="convert_vlm_output_to_jsonl.log",
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    logger = logging.getLogger(__name__)

    input_jsonl_path = config_loader.get_input_folder()
    output_jsonl_path = config_loader.get_output_folder()

    try:
        
        # Run conversion
        num_converted = convert_vlm_to_structure(input_jsonl_path, output_jsonl_path, logger)

        if num_converted > 0:
            logger.info(f"✓ Successfully converted {num_converted} records")
        else:
            logger.warning("No records were converted!")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)

    finally:
        logger.info("Convertion script finished")
 

if __name__ == "__main__":
    import sys

    # Configure paths for input and output
    config_path = "src/visual_grounding/config/entity_segmentation_config.yaml"

    main(config_path)