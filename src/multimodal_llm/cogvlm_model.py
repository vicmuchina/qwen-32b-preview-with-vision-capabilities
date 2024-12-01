"""
CogVLM Model Implementation for Vision-Language Processing

This module implements the CogVLM model integration for multimodal processing,
enabling vision-language understanding and generation tasks. It provides a unified
interface for processing both images and text inputs.

Features:
    - Automatic device detection (CPU/CUDA)
    - Float16 precision for optimal performance
    - Integrated image processing capabilities
    - Tokenization with automatic padding handling
    - Batch processing support for multiple inputs

Dependencies:
    - PyTorch
    - Transformers
    - PIL (Python Imaging Library)

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import os
import logging
from typing import Union, Optional, List, Tuple
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, ProcessorMixin
from . import config

# Configure logging
logger = logging.getLogger(__name__)

class CogVLMProcessor:
    """
    A processor class for handling vision-language tasks using the CogVLM model.
    
    This class provides methods for:
    - Model initialization and configuration
    - Image and text processing
    - Multimodal inference
    - Resource management and optimization
    
    Attributes:
        device (str): The computing device (cuda/cpu)
        torch_type: The PyTorch data type for computation
        model: The main CogVLM model instance
        processor: The image processor component
        tokenizer: The text tokenizer component
    """
    def __init__(self):
        """Initialize CogVLM model for visual processing."""
        logger.info("Initializing CogVLM2 Model...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_type = torch.float16  # Use float16 for better compatibility
        
        # Load model components
        logger.info("Loading model components...")
        
        # Initialize model first
        self.model = AutoModelForCausalLM.from_pretrained(
            config.COGVLM_MODEL_ID,
            torch_dtype=self.torch_type,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        
        # Initialize processor first
        self.processor = AutoProcessor.from_pretrained(
            config.COGVLM_MODEL_ID,
            trust_remote_code=True
        )
        
        # Initialize tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.COGVLM_MODEL_ID,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added [PAD] token")
                # Resize model embeddings to account for new token
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model loaded successfully!")
        self.history = []

    def process_query(self, query: str, image_path: str) -> str:
        """Process a query about an image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Resize image if needed (CogVLM expects specific dimensions)
            target_size = (224, 224)  # Standard size for many vision models
            if image.size != target_size:
                logger.info(f"Resizing image from {image.size} to {target_size}")
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Format conversation for the model
            conversation = []
            for q, a in self.history:
                conversation.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ])
            conversation.append({"role": "user", "content": query})
            
            # Prepare the prompt
            prompt = ""
            for msg in conversation:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt += f"Human: {content}\n"
                else:
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant: "
            
            # Process image
            vision_inputs = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )
            
            # Process text without padding
            text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            
            # Combine inputs
            inputs = {
                **vision_inputs,
                **text_inputs
            }
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Get the length of input tokens
                input_length = inputs['input_ids'].shape[1]
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                ).strip()
            
            # Update history
            self.history.append((query, response))
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def __call__(self, *args, **kwargs) -> str:
        """Convenience method to call process_query."""
        return self.process_query(*args, **kwargs)
