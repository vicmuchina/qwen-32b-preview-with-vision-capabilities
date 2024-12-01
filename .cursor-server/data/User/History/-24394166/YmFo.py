"""
CogVLM2 Model Implementation for Vision-Language Processing

This module implements the CogVLM2 model integration for multimodal processing,
providing advanced vision-language understanding capabilities.

Features:
    - Int4 quantization for efficient memory usage
    - Support for 8K content length
    - High resolution image processing (up to 1344x1344)
    - Multilingual support (English and Chinese)
"""

import os
import logging
from typing import Union, Optional, List, Tuple
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import config

# Configure logging
logger = logging.getLogger(__name__)

class CogVLM2Processor:
    """
    A processor class for handling vision-language tasks using the CogVLM2 model.
    
    This class provides methods for:
    - Model initialization and configuration
    - Image and text processing
    - Multimodal inference
    - Resource management and optimization
    
    Attributes:
        device (str): The computing device (cuda/cpu)
        torch_type: The PyTorch data type for computation
        model: The main CogVLM2 model instance
        tokenizer: The text tokenizer component
    """
    def __init__(self):
        """Initialize CogVLM2 model for visual processing."""
        logger.info("Initializing CogVLM2 Model...")
        
        self.model_name = "THUDM/cogvlm2-llama3-chat-19B-int4"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
        
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
