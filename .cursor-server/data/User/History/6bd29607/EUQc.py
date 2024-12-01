"""
QwQ-32B-Preview Model Implementation

This module implements the QwQ-32B-Preview model integration,
providing advanced language understanding and generation capabilities.

Features:
    - 32.5B parameters
    - 64 layers with 40 attention heads for Q and 8 for KV
    - 32,768 token context length
    - Multilingual support (English and Chinese)
"""

import os
import logging
from typing import List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import config

class QwQProcessor:
    def __init__(self):
        """Initialize QwQ-32B-Preview model."""
        self.model_name = "Qwen/QwQ-32B-Preview"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using QwQ model."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
