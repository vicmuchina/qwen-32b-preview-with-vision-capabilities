"""
Context Management System for Multimodal Interactions

This module implements a sophisticated context management system for handling
multimodal (vision and language) interactions. It maintains the state and
relationships between different visual elements and their textual descriptions.

Key Components:
    - ConfidenceLevel: Enumeration of confidence scores
    - DetailRequest: Structure for requesting specific visual details
    - VisualDetail: Container for visual information and metadata
    - ContextManager: Main class managing the context hierarchy

Features:
    - Hierarchical context management
    - Confidence level tracking
    - Relationship mapping between visual elements
    - Missing detail tracking
    - Temporal context maintenance

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ConfidenceLevel(Enum):
    """
    Enumeration of confidence levels for visual analysis results.
    
    Levels:
        HIGH: High confidence in the analysis
        MEDIUM: Moderate confidence in the analysis
        LOW: Low confidence, may need verification
        UNCERTAIN: Results are uncertain or ambiguous
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class DetailRequest:
    """
    Data class for requesting specific visual details from an image.
    
    Attributes:
        query (str): The specific detail being requested
        image_idx (int): Index of the target image
        region (Optional[str]): Specific region in the image
        aspect (str): Aspect of the image to analyze
        confidence_required (float): Minimum confidence threshold
    """
    query: str
    image_idx: int
    region: Optional[str]
    aspect: str
    confidence_required: float

@dataclass
class VisualDetail:
    """
    Data class containing detailed visual information and metadata.
    
    Attributes:
        content (str): The actual visual information
        confidence (float): Confidence score of the analysis
        source_image (int): Source image index
        region (Optional[str]): Region in the image
        timestamp (float): Time of analysis
        is_verified (bool): Verification status
    """
    content: str
    confidence: float
    source_image: int
    region: Optional[str]
    timestamp: float
    is_verified: bool = False

class ContextManager:
    """
    Main class managing the context hierarchy for multimodal interactions.
    
    Attributes:
        visual_contexts (Dict): Image key -> Initial context
        detail_contexts (Dict): Image key -> {detail_key -> VisualDetail}
        relationship_graph (Dict): Track relationships between details
        missing_details (Set): Track details that couldn't be found
    """
    def __init__(self):
        self.visual_contexts = {}  # Image key -> Initial context
        self.detail_contexts = {}  # Image key -> {detail_key -> VisualDetail}
        self.relationship_graph = {}  # Track relationships between details
        self.missing_details = set()  # Track details that couldn't be found
        
    def add_initial_context(self, image_key: str, context: str, confidence: float):
        """
        Add initial visual context for an image.
        
        Args:
            image_key (str): Unique key for the image
            context (str): Initial context for the image
            confidence (float): Confidence score for the context
        """
        self.visual_contexts[image_key] = {
            'content': context,
            'confidence': confidence,
            'details_requested': set()
        }
        
    def add_detail_context(self, image_key: str, detail_key: str, detail: VisualDetail):
        """
        Add detailed analysis for a specific aspect.
        
        Args:
            image_key (str): Unique key for the image
            detail_key (str): Unique key for the detail
            detail (VisualDetail): Detailed analysis for the aspect
        """
        if image_key not in self.detail_contexts:
            self.detail_contexts[image_key] = {}
        
        self.detail_contexts[image_key][detail_key] = detail
        
        # Update relationship graph for high-confidence details
        if detail.confidence > 0.7:
            if detail_key not in self.relationship_graph:
                self.relationship_graph[detail_key] = set()
            
            # Link to other details from same image
            for other_key in self.detail_contexts[image_key]:
                if other_key != detail_key:
                    self.relationship_graph[detail_key].add(other_key)
                    
    def get_missing_details(self) -> List[str]:
        """
        Get list of details that were requested but not found.
        
        Returns:
            List[str]: List of missing details
        """
        return list(self.missing_details)
    
    def mark_detail_missing(self, detail_request: DetailRequest):
        """
        Mark a detail as missing after failed attempts to find it.
        
        Args:
            detail_request (DetailRequest): Request for the missing detail
        """
        detail_key = f"Image {detail_request.image_idx + 1}: {detail_request.aspect}"
        self.missing_details.add(detail_key)
    
    def get_context_summary(self, image_keys: List[str]) -> Dict:
        """
        Get a summary of all context for specified images.
        
        Args:
            image_keys (List[str]): List of image keys
        
        Returns:
            Dict: Summary of context for the images
        """
        summary = {
            'initial_contexts': {},
            'detailed_contexts': {},
            'missing_details': list(self.missing_details),
            'relationships': self.relationship_graph
        }
        
        for image_key in image_keys:
            if image_key in self.visual_contexts:
                summary['initial_contexts'][image_key] = self.visual_contexts[image_key]
            if image_key in self.detail_contexts:
                summary['detailed_contexts'][image_key] = {
                    k: v.__dict__ for k, v in self.detail_contexts[image_key].items()
                }
                
        return summary
    
    def evaluate_context_completeness(self, query: str) -> tuple[float, List[str]]:
        """
        Evaluate how complete our context is for answering the query.
        
        Args:
            query (str): Query to evaluate context completeness for
        
        Returns:
            tuple[float, List[str]]: Average confidence score and list of missing aspects
        """
        total_confidence = 0
        missing_aspects = []
        relevant_details = 0
        
        # Process initial and detailed contexts
        for image_data in self.visual_contexts.values():
            total_confidence += image_data['confidence']
            relevant_details += 1
        
        for image_details in self.detail_contexts.values():
            for detail in image_details.values():
                if detail.confidence > 0:
                    total_confidence += detail.confidence
                    relevant_details += 1
                else:
                    missing_aspects.append(f"Missing or low confidence: {detail}")
        
        avg_confidence = total_confidence / max(relevant_details, 1)
        return avg_confidence, missing_aspects
    
    def get_related_details(self, detail_key: str) -> List[str]:
        """
        Get related details based on the relationship graph.
        
        Args:
            detail_key (str): Key of the detail to find related details for
        
        Returns:
            List[str]: List of related details
        """
        return list(self.relationship_graph.get(detail_key, set()))
    
    def clear(self):
        """
        Clear all stored contexts and relationships.
        """
        self.visual_contexts.clear()
        self.detail_contexts.clear()
        self.relationship_graph.clear()
        self.missing_details.clear()
