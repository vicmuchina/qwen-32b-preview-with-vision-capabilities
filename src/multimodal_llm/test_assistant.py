"""Test suite for the Multimodal AI Assistant."""
import unittest
import os
import tempfile
from PIL import Image
import numpy as np
from multimodal_llm.interactive_chat import MultimodalAssistant

class TestMultimodalAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.assistant = MultimodalAssistant()
        
        # Create a test image
        cls.test_image_path = os.path.join(tempfile.gettempdir(), 'test_image.png')
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(cls.test_image_path)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

    def setUp(self):
        """Reset assistant state before each test."""
        self.assistant.clear_history()
        self.assistant.clear_image()

    def test_text_only_conversation(self):
        """Test basic text conversation without images."""
        # Test simple query
        response = self.assistant.process_query("What is 2+2?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test conversation history
        self.assertEqual(len(self.assistant.chat_history), 1)
        self.assertEqual(self.assistant.chat_history[0][0], "What is 2+2?")
        
        # Test multiple turns
        self.assistant.process_query("Who invented Python?")
        self.assertEqual(len(self.assistant.chat_history), 2)

    def test_image_loading(self):
        """Test image loading functionality."""
        # Test valid image
        response = self.assistant.load_image(self.test_image_path)
        self.assertIn("Successfully loaded image", response)
        self.assertEqual(self.assistant.current_image, self.test_image_path)
        
        # Test invalid image path
        response = self.assistant.load_image("/nonexistent/path.jpg")
        self.assertIn("Error", response)
        
        # Test invalid image file
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            f.write(b'not an image')
            f.flush()
            response = self.assistant.load_image(f.name)
            self.assertIn("Error", response)

    def test_visual_queries(self):
        """Test image-based queries."""
        # Load test image
        self.assistant.load_image(self.test_image_path)
        
        # Test visual query
        response = self.assistant.process_query("What do you see in this image?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test non-visual query with image loaded
        response = self.assistant.process_query("What is the capital of France?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_conversation_management(self):
        """Test conversation management features."""
        # Test history tracking
        queries = [
            "Hello!",
            "How are you?",
            "What's the weather like?"
        ]
        
        for query in queries:
            self.assistant.process_query(query)
            
        self.assertEqual(len(self.assistant.chat_history), len(queries))
        
        # Test history clearing
        self.assistant.clear_history()
        self.assertEqual(len(self.assistant.chat_history), 0)
        
        # Test stats
        self.assistant.process_query("Test query")
        stats = self.assistant.get_conversation_stats()
        self.assertEqual(stats['total_turns'], 1)
        self.assertGreater(stats['total_tokens'], 0)
        self.assertFalse(stats['has_image'])

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test empty query
        response = self.assistant.process_query("")
        self.assertIn("Error", response.lower())
        
        # Test very long query
        long_query = "test " * 1000
        response = self.assistant.process_query(long_query)
        self.assertIsInstance(response, str)
        
        # Test with invalid image path
        self.assistant.current_image = "/nonexistent/path.jpg"
        response = self.assistant.process_query("What's in this image?")
        self.assertIn("Error", response)

def run_manual_test():
    """Run interactive manual test."""
    print("\nRunning manual test session...")
    assistant = MultimodalAssistant()
    
    # Test text-only conversation
    print("\n1. Testing text-only conversation:")
    response = assistant.process_query("What is artificial intelligence?")
    print(f"Q: What is artificial intelligence?\nA: {response}")
    
    # Test image loading (if test image exists)
    print("\n2. Testing image loading:")
    test_image = os.path.join(tempfile.gettempdir(), 'test_image.png')
    if not os.path.exists(test_image):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(test_image)
    
    response = assistant.load_image(test_image)
    print(f"Loading image response: {response}")
    
    # Test visual query
    print("\n3. Testing visual query:")
    response = assistant.process_query("What do you see in this image?")
    print(f"Q: What do you see in this image?\nA: {response}")
    
    # Test conversation stats
    print("\n4. Testing conversation stats:")
    stats = assistant.get_conversation_stats()
    print("Conversation statistics:", stats)
    
    # Cleanup
    if os.path.exists(test_image):
        os.remove(test_image)
    
    print("\nManual test completed!")

if __name__ == '__main__':
    # Run automated tests
    print("Running automated tests...")
    unittest.main(argv=[''], exit=False)
    
    # Run manual test
    run_manual_test()
