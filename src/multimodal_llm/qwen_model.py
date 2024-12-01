"""
Qwen Model Implementation Module

This module provides a wrapper for the Qwen-32B language model with optimized quantization
and inference capabilities. It includes functionality for model loading, text generation,
and streaming responses.

Key Features:
    - Supports both 4-bit and 8-bit quantization
    - Implements efficient text generation with customizable parameters
    - Provides streaming capability for real-time response generation
    - Handles model loading with automatic device mapping

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict, Optional, Union, Generator
import time
from threading import Thread

# Import configurations
QWEN_MODEL_NAME = "Qwen/QwQ-32B-Preview"
DEVICE_MAP = "auto"
COMPUTE_DTYPE = "torch.float16"

# Quantization settings
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "load_in_8bit": False
}

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9

class QwenProcessor:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize Qwen model with quantization."""
        self.model_name = model_name or QWEN_MODEL_NAME
        self.device_map = device or DEVICE_MAP
        
        try:
            print(f"Loading Qwen model: {self.model_name}")
            print(f"Device map: {self.device_map}")
            print(f"Quantization mode: {'8-bit' if QUANTIZATION_CONFIG.get('load_in_8bit') else '4-bit'}")
            
            # Configure quantization
            if QUANTIZATION_CONFIG.get("load_in_8bit", False):
                # 8-bit quantization
                quantization_config = {
                    "load_in_8bit": True,
                    "llm_int8_threshold": QUANTIZATION_CONFIG.get("llm_int8_threshold", 6.0),
                    "llm_int8_skip_modules": QUANTIZATION_CONFIG.get("llm_int8_skip_modules"),
                    "llm_int8_enable_fp32_cpu_offload": QUANTIZATION_CONFIG.get("llm_int8_enable_fp32_cpu_offload", False)
                }
            else:
                # 4-bit quantization
                quantization_config = {
                    "load_in_4bit": QUANTIZATION_CONFIG["load_in_4bit"],
                    "bnb_4bit_compute_dtype": QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"],
                    "bnb_4bit_quant_type": QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
                    "bnb_4bit_use_double_quant": QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
                    "torch_dtype": getattr(torch, COMPUTE_DTYPE.split('.')[-1]),
                    "device_map": self.device_map,
                    "trust_remote_code": True
                }
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **quantization_config
            ).eval()
            
            # Print memory usage
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"Error loading Qwen model: {str(e)}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the Qwen model."""
        try:
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

            if stream:
                # Set up streamer
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                generation_kwargs["streamer"] = streamer

                # Start generation in a separate thread
                inputs.update(generation_kwargs)
                thread = Thread(target=self.model.generate, kwargs=inputs)
                thread.start()

                # Stream the output
                response = ""
                for text in streamer:
                    response += text
                    yield text

                return
            
            # Non-streaming generation
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """Chat with the model using a message format."""
        try:
            # Format chat prompt
            formatted_prompt = ""
            if system_prompt:
                formatted_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            formatted_prompt += "<|im_start|>assistant\n"
            
            # Generate response
            return self.generate_response(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error: {str(e)}"
