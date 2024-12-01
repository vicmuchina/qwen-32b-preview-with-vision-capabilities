"""
Configuration Settings for Multimodal LLM System

This module contains all configuration parameters for the multimodal language model
system, including model settings, paths, and runtime parameters.

Configuration Categories:
    - Base Paths: Directory structure and file locations
    - Model Configurations: Model IDs and versions
    - Model Parameters: Sequence lengths and sizes
    - Conversation Settings: History and context management
    - Inference Settings: Generation parameters
    - Confidence Thresholds: Quality control parameters
    - Memory Management: Device mapping and data types
    - Quantization: Model compression settings

Notes:
    - All paths are relative to the project root directory
    - Memory settings are optimized for typical GPU configurations
    - Quantization settings balance quality and memory usage

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model configurations
QWEN_MODELS = {
    "32b": "Qwen/Qwen-32B",        # Full 32B parameter model
    "32b_preview": "Qwen/QwQ-32B-Preview",  # Preview version with optimizations
    "14b": "Qwen/Qwen-14B",        # Medium-sized model
    "7b": "Qwen/Qwen-7B",          # Smaller model for limited resources
    "1.8b": "Qwen/Qwen-1_8B"       # Lightweight model for testing
}

QWEN_MODEL_NAME = "Qwen/QwQ-32B-Preview"  # Using latest preview for best performance
COGVLM_MODEL_ID = "THUDM/cogvlm2-llama3-chat-19B-int4"  # Vision model with int4 quantization
COGVLM_REVISION = "main"

# Model parameters - Optimized for maximum context handling
MAX_SEQUENCE_LENGTH = 32768  # Maximum input context length
MAX_NEW_TOKENS = 32768      # Maximum generation length
COGVLM_IMAGE_SIZE = 1344    # Optimal image resolution for vision model

# Conversation management
MAX_HISTORY_LENGTH = 20     # Number of conversation turns to retain
MAX_TOKENS_PER_TURN = 512   # Token limit per response for memory efficiency
SLIDING_WINDOW_SIZE = 10    # Recent context window size

# Inference settings
TEMPERATURE = 0.7
TOP_P = 0.9

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for keeping visual details

# Memory management
DEVICE_MAP = "auto"  # Let transformers handle memory mapping
COMPUTE_DTYPE = torch.float16
MAX_VISUAL_CONTEXTS = 5  # Maximum number of image contexts to keep

# Quantization configurations
QUANTIZATION_CONFIGS = {
    "4bit": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",  # Using nf4 for better quality
        "bnb_4bit_use_double_quant": True,  # Enable double quantization for more memory savings
        "load_in_8bit": False
    },
    "8bit": {
        "load_in_4bit": False,
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": None,
        "llm_int8_enable_fp32_cpu_offload": True  # Enable CPU offloading if needed
    }
}

# Use 4-bit for 24GB RAM setup
QUANTIZATION_CONFIG = QUANTIZATION_CONFIGS["4bit"]

# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant with both visual and language understanding capabilities.
You can see and analyze images, and engage in natural conversations about them.
You maintain context of the conversation while being mindful of memory limitations."""

QWEN_SYSTEM_PROMPT = """You are a helpful AI assistant that can understand both text and images. 
You aim to provide accurate, informative, and engaging responses while maintaining a natural conversational style.
For image-related queries, you'll describe what you see and answer questions about the visual content.
For text-only queries, you'll provide knowledgeable responses based on your training.
Keep responses concise but informative."""

QWEN_PROMPT_TEMPLATE = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{visual_context}
{query}<|im_end|>
<|im_start|>assistant"""

COGVLM_INITIAL_PROMPT = """Please analyze this image carefully and provide a detailed description. Focus on key elements, their relationships, and any notable features."""

COGVLM_DETAIL_PROMPT = """Focus specifically on this detail: {detail_request}

Please provide:
1. A precise description of what you observe
2. A confidence score for your observation [score: X.XX]
3. If you cannot find or identify the requested detail, explicitly state:
   "Detail not found: [reason]" with [score: 0.0]"""

COGVLM_VERIFICATION_PROMPT = """Please verify the following analysis:
{analysis}

Evaluate each statement and provide:
1. Confidence score [score: X.XX]
2. Verification status [CONFIRMED/UNCERTAIN/INCORRECT]
3. Any corrections needed"""

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant based on the QwQ-32B-Preview model. You aim to provide accurate, helpful, and safe responses. You should:
1. Be direct and clear in your responses
2. Acknowledge your capabilities and limitations
3. Maintain a friendly and professional tone"""

# Generation settings
GENERATION_CONFIG = {
    "do_sample": True,  # Enable sampling
    "temperature": 0.7,  # Control randomness (lower = more deterministic)
    "top_p": 0.9,  # Nucleus sampling parameter
    "max_new_tokens": 512,  # Maximum length of generated text
    "pad_token_id": 0,  # Padding token ID
    "eos_token_id": 2,  # End of sequence token ID
}

# Memory optimization settings
ATTENTION_SLICING = True  # Enable attention slicing for memory efficiency
GRADIENT_CHECKPOINTING = False  # Disable since we're not training
CPU_OFFLOADING = False  # Keep disabled unless absolutely necessary
