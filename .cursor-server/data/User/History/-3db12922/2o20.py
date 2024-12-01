"""
Configuration settings for the multimodal LLM system.
"""

import os

# Model paths and settings
CACHE_DIR = os.path.expanduser("~/.cache/multimodal_llm/models")

# Chat settings
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in chat history
MAX_NEW_TOKENS = 512     # Maximum number of tokens to generate in responses

# CogVLM2 settings
COGVLM_MODEL_ID = "THUDM/cogvlm2-llama3-chat-19B-int4"
COGVLM_MAX_IMAGE_SIZE = 1344  # Maximum image size (both width and height)

# QwQ settings
QWQ_MODEL_ID = "Qwen/QwQ-32B-Preview"
QWQ_CONTEXT_LENGTH = 32768  # Maximum context length for QwQ model

# System prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful and harmless assistant. You are a multimodal AI that can understand both text and images. You should think step-by-step and provide clear, accurate responses."""
