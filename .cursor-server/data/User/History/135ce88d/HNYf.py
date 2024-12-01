"""Script to pre-download models to local cache."""
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Pre-download all required models to local cache."""
    cache_dir = os.path.expanduser("~/.cache/multimodal_llm/models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download CogVLM2
    logger.info("Downloading CogVLM2 model...")
    model_name_cogvlm = "THUDM/cogvlm2-llama3-chat-19B-int4"
    AutoTokenizer.from_pretrained(model_name_cogvlm, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        model_name_cogvlm,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Download QwQ-32B
    logger.info("Downloading QwQ-32B-Preview model...")
    model_name_qwen = "Qwen/QwQ-32B-Preview"
    AutoTokenizer.from_pretrained(model_name_qwen)
    AutoModelForCausalLM.from_pretrained(
        model_name_qwen,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
