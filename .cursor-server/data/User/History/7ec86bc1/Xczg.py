import torch
from transformers import AutoModelForCausalLM, AutoConfig

class CogVLM2Processor:
    def __init__(self):
        model_name = "THUDM/cogvlm-chat-hf"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Initialize the model with GPU support and quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # This will automatically handle GPU placement
            load_in_4bit=True,  # Enable 4-bit quantization
            torch_dtype=torch.float16  # Use half precision
        )
        
        # ... rest of your initialization ... 