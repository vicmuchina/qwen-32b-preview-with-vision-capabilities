"""
Interactive Chat Interface for Multimodal LLM

This module provides an interactive command-line interface for interacting with
the multimodal language model system. It combines vision capabilities from CogVLM2
and language processing from QwQ to create a comprehensive chat experience.
"""

import os
import sys
import logging
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multimodal_llm.multimodal_pipeline import MultimodalPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveChat:
    """Interactive chat interface for the multimodal system."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.pipeline = MultimodalPipeline()
        self.current_image = None
        
    def run(self):
        """Run the interactive chat loop."""
        print("Welcome to the Multimodal Chat Interface!")
        print("Commands:")
        print("  /image <path> : Load an image")
        print("  /clear : Clear current image")
        print("  /exit : Exit the chat")
        print("\nReady for chat!")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == '/exit':
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == '/clear':
                    self.current_image = None
                    print("Image cleared.")
                    continue
                    
                elif user_input.lower().startswith('/image '):
                    image_path = user_input[7:].strip()
                    if os.path.exists(image_path):
                        self.current_image = image_path
                        print(f"Image loaded: {image_path}")
                        # Get initial image description
                        response = self.pipeline.process_input("Describe this image.", image_path)
                        print("\nInitial image analysis:")
                        print(response)
                    else:
                        print(f"Error: Image file not found: {image_path}")
                    continue
                
                if not user_input:
                    continue
                
                # Process the input
                response = self.pipeline.process_input(user_input, self.current_image)
                print("\nAssistant:", response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive chat with multimodal LLM")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        chat = InteractiveChat()
        chat.run()
    else:
        print("Please use --interactive flag to start chat")

if __name__ == "__main__":
    main()
