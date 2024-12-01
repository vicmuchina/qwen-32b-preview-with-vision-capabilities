import torch
from transformers import AutoModelForCausalLM, AutoConfig

class CogVLM2Processor:
    def __init__(self):
        model_name = "THUDM/cogvlm-chat-hf"  # Replace with your actual model name
        model_name = "your-model-name"  # Replace with your actual model name
        
        # Load the model configuration
        config = AutoConfig.from_pretrained(model_name)
        
        # Disable quantization by setting quantization_config to None
        config.quantization_config = None
        
        # Initialize the model without quantization and load on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map='cpu',
            torch_dtype=torch.float32,
            # ... any other existing parameters ...
        )
        
        # ... rest of your initialization ... 