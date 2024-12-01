import torch
from transformers import AutoModelForCausalLM, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CogVLM2Processor:
    def __init__(self):
        model_name = "THUDM/cogvlm-chat-hf"
        
        logger.info("Starting model initialization...")
        logger.info("Checking device availability...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        logger.info("Loading model weights (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            # Add progress bar
            use_progress_bar=True
        )
        
        logger.info("Model initialization complete!")
        
        # ... rest of your initialization ... 