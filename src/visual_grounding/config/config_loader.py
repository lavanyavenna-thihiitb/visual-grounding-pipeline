import yaml
import os

class ConfigLoader:
    def __init__(self, config_path):
        config_path = os.path.abspath(config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) or {}
        self.model_type = self.config.get("model_type") or (
            self.config.get("models", [{}])[0].get("type")
            if self.config.get("models") else None
        )
        if not self.model_type:
            raise ValueError("model_type not set and no models[] with a 'type' found.")

    def get_model_type(self):
        return self.model_type

    def get_input_folder(self):
        return self.config["paths"]["input_folder"]

    def get_cuda_devices(self):
        return self.config.get("CUDA_DEVICES")

    def get_batch_size(self):
        return self.config.get("batch_size")

    def get_output_folder(self):
        return self.config["paths"]["output_folder"]

    def get_prompt_path(self):
        return self.config["paths"]["prompt_path"]

    def get_all_models(self):
        return self.config.get("models", [])

    def get_model_by_type(self, model_type=None):
        if model_type is None:
            model_type = self.model_type
        for m in self.get_all_models():
            if m.get("type") == model_type:
                return m
        raise ValueError(f"Model type '{model_type}' not found in config.")

    def get_model_name(self, model_type=None):
        return self.get_model_by_type(model_type).get("name")

    def get_model_params(self, model_type=None):
        return self.get_model_by_type(model_type).get("params", {}) or {}

    def get_sampling_params(self, model_type=None):
        return self.get_model_by_type(model_type).get("sampling_params", {}) or {}

    # Use for vLLM's not for huggingface 
    def get_llm_kwargs(self, model_type=None, include_model=True):
      
        params = dict(self.get_model_params(model_type))  
        if "limit_mm_per_prompt" in params:
            v = params["limit_mm_per_prompt"]
            if v is None:
                params.pop("limit_mm_per_prompt", None)
            elif isinstance(v, int):
                params["limit_mm_per_prompt"] = {"image": v}

        if include_model:
            params["model_name"] = self.get_model_name(model_type)

        return {k: v for k, v in params.items() if v is not None}

    def get_sampling_params_kwargs(self, model_type=None):
        
        sp = self.get_sampling_params(model_type)
        return {k: v for k, v in sp.items() if v is not None}


