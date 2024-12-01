"""
Interactive Chat Interface Module

This module provides a command-line interface for interacting with the
multimodal model, supporting both text and image inputs.

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InteractiveChat:
    """
    Interactive chat interface for handling multimodal conversations.
    Supports commands for loading images and managing chat context.
    """

    def __init__(self):
        """
        Initialize the chat interface and set up the multimodal pipeline.
        Provides progress feedback during initialization.
        """
        logger.info("Initializing chat interface...")
        print("Setting up model (this may take a few minutes)...")
        self.pipeline = MultimodalPipeline()
        print("Setup complete! Ready for chat.")