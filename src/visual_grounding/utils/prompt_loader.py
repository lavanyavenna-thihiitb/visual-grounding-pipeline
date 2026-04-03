import os
import yaml
import logging

logger = logging.getLogger(__name__)


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

def load_prompt_for_entity_segmentation(prompt_path: str):
    pass