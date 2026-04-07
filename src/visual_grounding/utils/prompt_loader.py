import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_prompt_config(prompt_file: str) -> dict:
    """
    Load YAML prompt configuration.
    """
    with open(prompt_file, "r") as f:
        return yaml.safe_load(f)

def extract_prompt_files(prompt_path: str) -> dict:
    prompt_path = os.path.abspath(prompt_path)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Prompt file not found at: {prompt_path}"
        )
    
    with open(prompt_path, "r") as f:
        prompt = yaml.safe_load(f) or {}

    if "system" not in prompt or "user" not in prompt:
        raise ValueError(
            f"Prompt file at '{prompt_path}' must contain both "
            f"'system' and 'user' keys. Found keys: {list(prompt.keys())}"
        )

    logger.info("Prompt loaded from: %s", prompt_path)

    return prompt

def get_system_prompt_for_entity_extraction(prompt: dict) -> str:
    return prompt["system"]

def get_user_prompt_for_entity_extraction(prompt: dict, caption: str) -> str:
    """
    Returns the user template with the caption interpolated.
    """
    return prompt["user"].format(caption=caption)

def prompt_for_entity_extraction(prompt: dict, caption: str):

    system_prompt = get_system_prompt_for_entity_extraction(prompt=prompt)
    caption_prompt = get_user_prompt_for_entity_extraction(prompt=prompt, caption=caption)

    return system_prompt, caption_prompt

def format_prompt_for_entity_segmentation(entites:list[str], counts: dict[str, int], prompt_file: str) -> str:
    """
    Format multi-entity prompt using YAML templates.

    Args:
        entities: list of entity strings.
        counts: dict mapping entity -> count.
        prompt_file: path to YAML config.

    Returns:
        formatted multi-line prompt string. 
    """

    prompt_config = load_prompt_config(prompt_file)

    templates = prompt_config["prompt_templates"]
    seperator = prompt_config["multi_entity"]["seperator"]

    prompts = []

    for entity in entites:
        count = counts.get(entity, 1)

        if count == -1:
            template = templates["find_all"]["template"]
            prompts.append(template.format(entity=entity))

        elif count == 1:
            template = templates["find_one"]["template"]
            prompts.append(template.format(entity=entity))

        else:
            template = templates["find_exact"]["template"]
            prompts.append(template.format(entity=entity, count=count))

    return seperator.join(prompts)

    