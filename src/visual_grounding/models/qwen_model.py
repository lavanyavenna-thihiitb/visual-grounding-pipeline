import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
import logging
import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from visual_grounding.config.config_loader import ConfigLoader
from visual_grounding.utils.prompt_loader import extract_prompt_files, prompt_for_entity_extraction

logger = logging.getLogger(__name__)

class Qwen_EntityExtraction:

    def __init__(self, config_path: str) -> None:
        self.config_loader = ConfigLoader(config_path)
        self.model_type = self.config_loader.get_model_type()
        self.model_name = self.config_loader.get_model_name(self.model_type) 

        self.tokenizer: Any = None
        self.model: Any = None 
        self.prompt = extract_prompt_files(self.config_loader.get_prompt_path())
        self.load_model()

    def load_model(self) -> None:

        """
        Loads the Qwen2.5 tokenizer and model from HuggingFace.
        All parameters (torch_dtype, device_map, trust_remote_code) come from the config file
        """

        model_params = self.config_loader.get_model_params(self.model_type)

        logger.info("Loading tokenizer: %s", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **model_params)

        logger.info("Loading model: %s | params: %s", self.model_name, model_params)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_params
        )

        self.model.eval()
        logger.info("Model loaded and set to eval mode.")

    def generate_outputs(self, caption: str) -> str:
        """
        Takes a raw caption string, builds the chat prompt, runs inference, and returns the raw decoded string from Qwen.
        """

        system_prompt, user_prompt = prompt_for_entity_extraction(self.prompt, caption)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply Qwen's chat template
        # tokenize=False returns a plain string — we tokenise in the next step
        # add_generation_prompt=True appends the <|assistant|> token so Qwen
        # knows it should start generating its reply immediately.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenise and move to device
        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        # Get sampling params from config and run inference
        sp_kwargs = self.config_loader.get_sampling_params_kwargs(self.model_type)

        logger.info("Generating output for caption: %r", caption)
        logger.debug("Sampling params: %s", sp_kwargs)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **sp_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return ""
        
        # model.generate() returns the full sequence (prompt + reply). 
        # Slice off the prompt portion so we only decode Qwen's reply
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        raw_output = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        logger.debug("Raw output: %r", raw_output)
        return raw_output
    

# ---------------------------------------------------------------------------
# __main__ — smoke test for load_model() and generate_outputs()
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    config_path = "src/visual_grounding/config/entity_extraction_config.yaml"
    # caption = "a fluffy dog and a dangerous dog are fighting in the yard"
    caption = "the dog chases the cat and the dog runs"

    # ── Step 1: Load model ───────────────────────────────────────────────────
    logger.info("Initialising Qwen_EntityExtraction …")
    model = Qwen_EntityExtraction(config_path)
    logger.info("Model initialised successfully.")

    # ── Step 2: Run a single caption through generate_outputs() ─────────────
    logger.info("Running generate_outputs() on caption: %r", caption)
    raw_output = model.generate_outputs(caption)

    # ── Step 3: Print result ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Caption : {caption}")
    print(f"Output  : {raw_output}")
    print("=" * 60)