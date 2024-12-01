"""Script to pre-download models to local cache."""
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cogvlm_model import CogVLMProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Pre-download all required models to local cache."""
    cache_dir = os.path.expanduser("~/.cache/multimodal_llm/models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download CogVLM
    logger.info("Downloading CogVLM model...")
    processor = CogVLMProcessor()
    
    # Download Qwen
    logger.info("Downloading Qwen model...")
    model_name = "Qwen/Qwen-72B-Chat"
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
