"""
Multimodal Pipeline Implementation

This module implements the pipeline for combining vision and language models,
coordinating between CogVLM2 for vision tasks and QwQ for language processing.
"""

import os
import logging
from typing import Optional, List, Dict, Union
from .cogvlm_model import CogVLM2Processor
from .qwen_model import QwQProcessor
from . import config

class MultimodalPipeline:
    """
    Pipeline for coordinating vision and language models.
    
    This class manages the interaction between:
    - CogVLM2 for vision processing
    - QwQ for language processing
    - Context management
    - Response generation
    """
    
    def __init__(self):
        """Initialize the multimodal pipeline."""
        self.vision_model = CogVLM2Processor()
        self.language_model = QwQProcessor()
        self.current_image = None
        
    def process_input(self, text: str, image_path: Optional[str] = None) -> str:
        """
        Process input which may include both text and image.
        
        Args:
            text (str): The text input/query
            image_path (Optional[str]): Path to an image file, if any
            
        Returns:
            str: Generated response
        """
        try:
            if image_path:
                # Process image with vision model
                vision_response = self.vision_model.process_image(image_path, text)
                
                # Combine vision insights with language model
                combined_prompt = f"Based on the image analysis: {vision_response}\n\nUser query: {text}"
                final_response = self.language_model.generate_response(
                    combined_prompt,
                    system_prompt=config.DEFAULT_SYSTEM_PROMPT
                )
                return final_response
            else:
                # Text-only processing
                return self.language_model.generate_response(
                    text,
                    system_prompt=config.DEFAULT_SYSTEM_PROMPT
                )
                
        except Exception as e:
            logging.error(f"Error in multimodal pipeline: {str(e)}")
            raise
