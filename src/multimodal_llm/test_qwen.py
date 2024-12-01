"""Test script for Qwen model."""
from qwen_model import QwenProcessor

def test_qwen_response():
    """Test Qwen model's response and word counting."""
    # Initialize model
    model = QwenProcessor()
    
    # Test prompt
    prompt = "How many r's are in the word strawberry?"
    
    # Generate response
    messages = [{"role": "user", "content": prompt}]
    response = model.chat(messages=messages)
    
    
    # Print results
    print("\nPrompt:", prompt)
    print("\nResponse:", response)
    print("\nWord count:", word_count)

if __name__ == "__main__":
    test_qwen_response()
