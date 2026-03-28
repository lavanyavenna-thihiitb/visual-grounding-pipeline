"""

Stage 1 of the Visual Grounding Pipeline.
Extracts entity phrases from a text caption using Qwen2.5.

"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from visual_grounding.samples.sample_captions import SAMPLE_CAPTIONS
from visual_grounding.prompts.entity_extraction_prompt import ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_TEMPLATE

log = logging.getLogger(__name__)

class EntityExtractor:
    """
    Loads a Qwen2.5 model and exposes an extract() method that takes a
    caption string and returns a structured dict of entities + count hints.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("EntityExtractor — device: %s", self.device)
        self.model_name = model_name

    def load_model(self):
        """
        Loads the Qwen2.5 tokeniser and model from HuggingFace.
        """
        log.info("Loading model from %s …", self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        log.info("Qwen model loaded and set to eval mode.")

    def _call_qwen(self, caption: str) -> str:
        """
        Builds the chat prompt from the caption, runs Qwen inference,
        and returns the raw decoded string.
        """

        messages = [
            {
                "role": "system",
                "content": ENTITY_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": ENTITY_EXTRACTION_USER_TEMPLATE.format(caption=caption),
            },
        ]

        # add_generation_prompt=True appends the <|assistant|> token so the
        # model knows it should start generating its reply immediately.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenise and move to device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,      # greedy — deterministic JSON output
                temperature=None,     # must be None when do_sample=False
                top_p=None,           # must be None when do_sample=False
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
 
        raw_output = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
 
        log.debug("Qwen raw output: %r", raw_output)
        return raw_output


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    extractor = EntityExtractor()

    for caption in SAMPLE_CAPTIONS:
        log.info(f"\nCaption: {caption}")
        result = extractor.extract(caption)
        log.info(f"Result:    {result}")