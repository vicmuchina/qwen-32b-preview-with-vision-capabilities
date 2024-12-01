"""Test script for CogVLM model."""
import os
from PIL import Image
import torch
from multimodal_llm.cogvlm_model import CogVLMProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cogvlm(image_path):
    """Run a simple test of CogVLM functionality using a specified image."""
    print("\nTesting CogVLM model...")
    
    try:
        # Initialize processor
        processor = CogVLMProcessor()
        
        # Print GPU info
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("No GPU available, using CPU")
        
        # Test basic image loading
        print("\nTesting image loading...")
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        # Test simple visual query
        print("\nTesting visual query...")
        test_query = "What is happening in this image?"
        print(f"Query: {test_query}")
        
        response = processor.process_query(test_query, image_path)
        print(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
    
    finally:
        print("\nTest completed!")


if __name__ == '__main__':
    image_path = "/teamspace/studios/this_studio/multimodal_llm/src/multimodal_llm/test_assets/Screenshot 2024-12-01 132254.png"
    test_cogvlm(image_path)
