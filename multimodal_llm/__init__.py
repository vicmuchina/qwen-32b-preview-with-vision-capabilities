"""
Multimodal LLM package initialization.

This package provides integration between language models and vision capabilities,
combining CogVLM and Qwen models for multimodal processing.

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

from .multimodal_pipeline import MultimodalPipeline
from .cogvlm_model import CogVLM2Processor
from .qwen_model import QwQProcessor

__all__ = ['MultimodalPipeline', 'CogVLM2Processor', 'QwQProcessor'] 