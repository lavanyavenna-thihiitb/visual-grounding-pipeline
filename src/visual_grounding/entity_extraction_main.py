import json
import re
import logging

from visual_grounding.models.qwen_model import Qwen_EntityExtraction

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Orchestrates Stage 1 of visual grounding pipeline.

    Takes a list of captions and returns a list of dicts of the form:
    {"entities": [...], "counts": {...}}

    Internally uses Qwen_EntityExtraction for inference and falls back to spaCy noun-phrase chunking if Qwen malformed JSON.

    Parameters
        config_path: str
            Path to entity_extraction_config.yaml
        spacy_model: str
            spacy model name used as fallback (default: en_core_web_sm)
    """

    def __init__(self, config_path: str) -> None:
        
        #Qwen Model
        self.model = Qwen_EntityExtraction(config_path)
        self.batch_size = self.model.config_loader.get_batch_size() or 30 #'or 30' means — if get_batch_size() returns None (key missing from YAML) then Pylance now knows self.batch_size is always an int and the error goes away.


    def _parse_output(self, raw_output: str) -> dict | None:
        """
        Parses the raw string returned by Qwen into a structured dict.

        Handles three real-world failure modes from Qwen:
            1. Output wrapped in markdown fences (```json ... ```)
            2. JSON preceed or followed by explanation prose
            3. COompletely malformed output - not JSON at all

        Returns a dict of the form:
            {"entities": [...], "counts": {...}}
        or None if parsing fails - caller should trigger spaCy fallback.
        """

        # Stip markdown code fences
        # Qwen often wraps output in ```json ... ``` even when told not to. We strip before attempting to parse.
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_output).strip()
        cleaned = cleaned.rstrip("`").strip()

        # Find teh first {...} block
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not brace_match:
            logger.warning("No JSON object found in Qwen Output: %r", raw_output)
            return None

        cleaned = brace_match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s | raw output: %r", e, raw_output)
            return None

        # Validate structure
        if not isinstance(parsed, dict):
            logger.warning("Parsed Json is not a dict: %r", parsed)
            return None

        if "entities" not in parsed or not isinstance(parsed["entities"], list):
            logger.warning("entities key missing or not a list: %r", parsed)
            return None

        if "counts" not in parsed or not isinstance(parsed["counts"], dict):
            logger.warning("'counts' key missing - auto-filling with 1 for all entities.")
            parsed["counts"] = {entity: 1 for entity in parsed["entities"]}

        # Step 5: Fill any missing count entries
        for entity in parsed["entities"]:
            if entity not in parsed["counts"]:
                parsed["counts"][entity] = 1

        return parsed 

    def extract(self, captions: list[str]) -> list[dict]:
        """
        Runs entity extraction on a list of captions in batches.

        For each caption:
            - Calls Qwen to get raw output
            - Parses raw output into {"entities": [...], "counts": {...}}
            - If parsing fails, logs a warning and appends an empty result

        Parameters
            captions: list[str]
            List of caption strings to extract entities from.

        Returns
            list[dict]
            One dict per caption, in the same order as the input list.
            Failed captions return {"entities": [], "counts": {}}
        """

        results = []
        total = len(captions)

        for batch_start in range(0, total, self.batch_size):
            batch = captions[batch_start: batch_start+self.batch_size]

            logger.info("Processing batch %d of captions", (batch_start // self.batch_size) + 1)

            for caption in batch:
                logger.info("Extracting entities for caption: %r", caption)

                # Step 1: get raw output from Qwen
                raw_output = self.model.generate_outputs(caption)

                # Step 2: parse raw output into structured dict
                parsed = self._parse_output(raw_output)

                # Handle parse failure
                if parsed is None:
                    logger.warning(
                        "Failed to parse Qwen output for caption: %r |"
                        "Raw output was: %r |"
                        "Returning empty result for this caption.",
                        caption, raw_output,
                    )

                    results.append({"entities":[], "counts": {}})

                else:
                    logger.info(
                        "Extracted %d entity/ entities: %s",
                        len(parsed["entities"]), parsed["entities"]
                    )

                    results.append(parsed)

        return results


if __name__ == "__main__":

    logging.basicConfig(
        filename='entity_extraction_1.log',
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )

    from visual_grounding.samples.sample_captions import SAMPLE_CAPTIONS

    config_path = 'src/visual_grounding/config/entity_extraction_config.yaml'

    # Initialise EntityExtractor
    logger.info("Initialising EnityExtractor ...")
    extractor = EntityExtractor(config_path)
    logger.info("EnityExtractor initialised successfully.")

    # Run all sample captions through extract()
    logger.info("Running extract() on %d captions...", len(SAMPLE_CAPTIONS))
    results = extractor.extract(SAMPLE_CAPTIONS)

    for caption, result in zip(SAMPLE_CAPTIONS, results):
        logger.info("\n" + "=" * 60)
        logger.info(f"Caption : {caption}")
        logger.info(f"Output  : {result}")
        logger.info("=" * 60)

