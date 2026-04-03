from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def load_test_jsonl(jsonl_path_str: str) -> list[dict]:
    """
    Reads the .jsonl file and returns a list of records.

    Expected format per line:
    {"image_path": "path/to/img.jpg", "captions":{"short_caption":"", "long_caption": "..."}}

    Skips and logs any malformed lines.
    """

    records = []
    jsonl_path = Path(jsonl_path_str)

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

            # Validate required keys
            if "image_path" not in record or "captions" not in record:
                logger.warning(
                    "Skipping line %d - missing 'image_path' or 'captions' key.", line_no
                )
                continue

            records.append(record)

        logger.info("Loaded %d valid records from %s", len(records), jsonl_path)

    return records
