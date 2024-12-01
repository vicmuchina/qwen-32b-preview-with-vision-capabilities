"""
Multimodal Pipeline Integration Module

This module provides a unified pipeline that integrates CogVLM's vision capabilities
with Qwen's language processing abilities. It manages the interaction between
visual and textual components while maintaining conversation context.

Key Components:
    - Vision Processing: Handles image analysis using CogVLM
    - Language Processing: Manages text generation using Qwen
    - Context Management: Maintains conversation history and state
    - Confidence Scoring: Evaluates response reliability

Features:
    - Seamless integration of vision and language models
    - Dynamic context management
    - Confidence-based response validation
    - Support for single and multiple image inputs
    - Iterative refinement of responses

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import logging
from typing import List, Dict, Optional, Union
from PIL import Image
import re
import time
from . import config
from .cogvlm_model import CogVLMProcessor
from .qwen_model import QwenProcessor
from .context_manager import (
    ContextManager,
    DetailRequest,
    VisualDetail,
    ConfidenceLevel
)

logger = logging.getLogger(__name__)

class MultimodalPipeline:
    """
    A pipeline class that orchestrates multimodal processing using vision and language models.
    
    This class coordinates the interaction between:
    - CogVLM for vision processing
    - Qwen for language generation
    - Context management for maintaining conversation state
    
    The pipeline supports:
    - Single and batch image processing
    - Context-aware conversation
    - Confidence-based response validation
    - Iterative response refinement
    
    Attributes:
        vision_processor (CogVLMProcessor): Handles image processing
        language_processor (QwenProcessor): Manages language generation
        context_manager (ContextManager): Maintains conversation context
        conversation_history (list): Stores interaction history
    """
    def __init__(self):
        """Initialize the multimodal pipeline."""
        self.vision_processor = CogVLMProcessor()
        self.language_processor = QwenProcessor()
        self.context_manager = ContextManager()
        self.conversation_history = []

    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from response."""
        matches = re.findall(r'\[score: (0\.\d+|\d\.\d+)\]', response)
        return float(matches[0]) if matches else 0.0

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to level."""
        if score >= config.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif score >= config.MIN_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        elif score > 0:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN

    def process_query(
        self,
        query: str,
        images: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
        max_iterations: int = 3
    ) -> str:
        """Process a query with context management."""
        images = [images] if images and not isinstance(images, list) else images or []

        # Initial analysis
        for idx, image in enumerate(images):
            image_key = str(image) if isinstance(image, str) else id(image)
            analysis = self.vision_processor.process_visual_query(image, config.COGVLM_INITIAL_PROMPT)
            confidence = self._extract_confidence_score(analysis)
            self.context_manager.add_initial_context(image_key, analysis, confidence)

        # Refinement loop
        iteration = 0
        previous_analysis = ""
        final_response = ""

        while iteration < max_iterations:
            # Get context summary
            image_keys = [str(img) if isinstance(img, str) else id(img) for img in images]
            context = self.context_manager.get_context_summary(image_keys)
            
            # Format prompt
            prompt_context = {
                "system_prompt": config.QWEN_SYSTEM_PROMPT,
                "initial_context": "\n".join(
                    f"Image {i+1}:\n{ctx['content']}"
                    for i, ctx in enumerate(context['initial_contexts'].values())
                ),
                "additional_context": "\n".join(
                    f"Image {i+1} Details:\n" + "\n".join(
                        f"- {k}: {v['content']}"
                        for k, v in img_details.items()
                    )
                    for i, img_details in enumerate(context['detailed_contexts'].values())
                ) if context['detailed_contexts'] else "",
                "missing_details": "\n".join(context['missing_details']),
                "query": query,
                "previous_analysis": previous_analysis
            }

            # Get Qwen response
            response = self.language_processor.generate_response(
                config.QWEN_PROMPT_TEMPLATE.format(**prompt_context)
            )

            # Process vision queries
            vision_queries = self._extract_vision_queries(response)
            if not vision_queries:
                # Verify final response
                verification = self.vision_processor.process_visual_query(
                    images[0],
                    config.COGVLM_VERIFICATION_PROMPT.format(analysis=response)
                )
                
                # Add confidence summary
                confidence, missing = self.context_manager.evaluate_context_completeness(query)
                confidence_level = self._get_confidence_level(confidence)
                final_response = (
                    f"{response}\n\n"
                    f"Confidence: {confidence_level.value.upper()}\n"
                    f"Missing: {', '.join(missing) if missing else 'None'}\n"
                    f"Verification: {verification}"
                )
                break

            # Process each vision query
            for query in vision_queries:
                image_idx, detail_request = self._parse_vision_query(query)
                if image_idx >= len(images):
                    continue
                    
                image = images[image_idx]
                image_key = str(image) if isinstance(image, str) else id(image)
                
                # Get detailed analysis
                detail_response = self.vision_processor.process_visual_query(
                    image,
                    config.COGVLM_DETAIL_PROMPT.format(detail_request=detail_request)
                )
                
                # Process detail
                confidence = self._extract_confidence_score(detail_response)
                detail = VisualDetail(
                    content=detail_response,
                    confidence=confidence,
                    source_image=image_idx,
                    region=None,
                    timestamp=time.time()
                )
                
                if confidence > 0:
                    self.context_manager.add_detail_context(
                        image_key,
                        detail_request.lower().strip(),
                        detail
                    )
                else:
                    self.context_manager.mark_detail_missing(
                        DetailRequest(
                            query=query,
                            image_idx=image_idx,
                            region=None,
                            aspect=detail_request,
                            confidence_required=config.MIN_CONFIDENCE_THRESHOLD
                        )
                    )
            
            previous_analysis = response
            iteration += 1

        # Update history
        self.conversation_history.append({
            "query": query,
            "images": [str(img) for img in images],
            "response": final_response,
            "iterations": iteration
        })

        return final_response

    def _extract_vision_queries(self, response: str) -> List[str]:
        """Extract vision queries from response."""
        return re.findall(r"<vision_query>(.*?)</vision_query>", response, re.DOTALL)

    def _parse_vision_query(self, query: str) -> tuple[int, str]:
        """Parse image reference and detail request."""
        image_idx = 0
        if matches := re.findall(r"Image[- ](\d+)", query, re.IGNORECASE):
            image_idx = int(matches[0]) - 1
        detail_request = re.sub(r"Image[- ](\d+)", "", query, flags=re.IGNORECASE).strip()
        return image_idx, detail_request

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history

    def clear_context(self):
        """Clear all contexts."""
        self.context_manager.clear()
