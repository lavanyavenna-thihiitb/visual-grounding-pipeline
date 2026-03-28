import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen_Model:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        
        # We declare these as Any to bypass the "BaseModelWithGenerate" type error
        self.tokenizer: Any = None
        self.model: Any = None
        
        self.load_model()

    def load_model(self) -> None:
        """Initializes the tokenizer and model using the source of truth config."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        self.model.eval()
        print("Model loaded and ready.")

    def generate(self, prompt: str, system_message: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> str:
        """Takes a user prompt and returns the model's response string."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # 1. Prepare the chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. Tokenize and move to the correct device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 3. Inference (Wrapped in no_grad for memory efficiency)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )

        # 4. Slice off the prompt IDs to get only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 5. Decode and return
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- Example Usage ---
if __name__ == "__main__":
    qwen = Qwen_Model()
    response = qwen.generate("Give me a short introduction to large language model.")
    print(f"\nQwen Response:\n{response}")