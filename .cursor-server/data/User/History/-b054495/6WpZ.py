"""
Interactive Chat Interface for Multimodal LLM

This module provides an interactive command-line interface for interacting with
the multimodal language model system. It combines vision capabilities from CogVLM
and language processing from Qwen to create a comprehensive chat experience.

Features:
    - Interactive command-line interface
    - Image loading and analysis
    - Hybrid vision-language query processing
    - Chat history management
    - Token usage tracking
    - Command system for user interaction

Usage:
    Run this module directly to start the interactive chat interface:
    ```
    python -m multimodal_llm.interactive_chat
    ```

Author: Victor Muchina
Email: vicmuchina1234@gmail.com
"""

import os
import sys
import logging
import argparse
from collections import deque
from src.multimodal_llm import config
from src.multimodal_llm.cogvlm_model import CogVLMProcessor
from src.multimodal_llm.qwen_model import QwenProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalAssistant:
    """
    Core assistant class that manages the multimodal interaction system.
    
    This class coordinates between:
    - CogVLM for vision processing
    - Qwen for language processing
    - Chat history management
    - Token usage tracking
    
    Attributes:
        cogvlm (CogVLMProcessor): Vision model processor
        qwen (QwenProcessor): Language model processor
        current_image (str): Path to currently loaded image
        chat_history (deque): Fixed-length queue of chat history
        total_tokens (int): Running count of tokens used
    """
    
    def __init__(self):
        """Initialize the assistant with both CogVLM and Qwen models."""
        self.cogvlm = CogVLMProcessor()
        self.qwen = QwenProcessor()
        self.current_image = None
        self.chat_history = deque(maxlen=config.MAX_HISTORY_LENGTH)
        self.total_tokens = 0
        
    def load_image(self, image_path: str) -> str:
        """
        Load and perform initial analysis of an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Initial description of the image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For other processing errors
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Store image path
            self.current_image = image_path
            
            # Initial analysis with CogVLM
            response = self.cogvlm.process_query("Describe this image.", image_path)
            return response
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            self.current_image = None
            raise
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using the appropriate model.
        
        Automatically selects between vision and language models based on
        whether an image is currently loaded.
        
        Args:
            query (str): User's input query
            
        Returns:
            str: Model's response to the query
        """
        try:
            if self.current_image:
                # Use CogVLM for image-related queries
                response = self.cogvlm.process_query(query, self.current_image)
            else:
                # Use Qwen for text-only queries with streaming
                messages = [{"role": "user", "content": query}]
                response = self.qwen.chat(messages, stream=True)
                
                # Handle streaming response
                if not isinstance(response, str):
                    full_response = ""
                    for text in response:
                        full_response += text
                    response = full_response
                
            # Update history
            self.chat_history.append({
                "query": query,
                "response": response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def clear_image(self):
        """Clear the current image."""
        self.current_image = None
    
    def get_conversation_stats(self) -> dict:
        """
        Get conversation statistics.
        
        Returns:
            dict: Dictionary containing conversation statistics
        """
        return {
            "total_turns": len(self.chat_history),
            "total_tokens": self.total_tokens,
            "average_tokens_per_turn": self.total_tokens / len(self.chat_history) if self.chat_history else 0,
            "has_image": self.current_image is not None
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history.clear()
        self.total_tokens = 0

class InteractiveChat:
    """
    Interactive chat interface class.
    
    This class manages the command-line interface and user interaction.
    
    Attributes:
        assistant (MultimodalAssistant): Multimodal assistant instance
    """
    
    def __init__(self):
        """Initialize chat interface."""
        self.assistant = MultimodalAssistant()
        
    def print_commands(self):
        """Print available commands."""
        print("Entering interactive mode. Commands:")
        print("/load [image_path] - Load a new image")
        print("/clear - Clear current image")
        print("/history - Show conversation history")
        print("/stats - Show conversation statistics")
        print("/reset - Clear conversation history")
        print("/exit - Exit the chat")
        
    def process_command(self, command: str) -> bool:
        """
        Process a command. Returns True if should continue, False if should exit.
        
        Args:
            command (str): User's input command
            
        Returns:
            bool: Whether to continue or exit
        """
        if command.startswith("/load "):
            image_path = command[6:].strip()
            try:
                response = self.assistant.load_image(image_path)
                print(f"Successfully loaded image: {image_path}")
                print("Initial analysis:", response)
            except Exception as e:
                print(f"Error loading image: {e}")
        elif command == "/clear":
            self.assistant.clear_image()
            print("Cleared current image")
        elif command == "/history":
            for i, entry in enumerate(self.assistant.chat_history):
                print(f"\nQ{i+1}: {entry['query']}")
                print(f"A{i+1}: {entry['response']}")
        elif command == "/stats":
            stats = self.assistant.get_conversation_stats()
            print(f"Total turns: {stats['total_turns']}")
            print(f"Current image: {self.assistant.current_image or 'None'}")
        elif command == "/reset":
            self.assistant.clear_history()
            print("Conversation history cleared")
        elif command == "/exit":
            return False
        return True
        
    def run(self):
        """Run the interactive chat interface."""
        self.print_commands()
        print("\nStart chatting!")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle commands
                if user_input.startswith("/"):
                    if not self.process_command(user_input):
                        break
                    continue
                
                # Process regular input
                if not user_input:
                    continue
                    
                print("\nAssistant:", end=" ", flush=True)
                
                if self.assistant.current_image:
                    # Use CogVLM for image-related queries
                    response = self.assistant.process_query(user_input)
                    print(response)
                else:
                    # Use Qwen for text-only queries with streaming
                    messages = [{"role": "user", "content": user_input}]
                    response = self.assistant.qwen.chat(messages, stream=True)
                    
                    if isinstance(response, str):
                        print(response)
                    else:
                        for text in response:
                            print(text, end="", flush=True)
                        print()
                    
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
