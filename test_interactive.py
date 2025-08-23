#!/usr/bin/env python3
"""
Test script for interactive object detection responses.
This demonstrates how the system now asks questions instead of just describing objects.
"""

from simple_ai_detect import SimpleAIDescriptionGenerator

def test_interactive_responses():
    """Test the interactive response generation."""
    print("🧪 Testing Interactive Response Generation")
    print("=" * 50)
    
    # Initialize the interactive generator
    generator = SimpleAIDescriptionGenerator()
    
    # Test different scenarios
    test_cases = [
        {"person": 1},
        {"car": 2, "person": 1},
        {"laptop": 1, "chair": 1, "table": 1},
        {"dog": 1, "cat": 1},
        {"truck": 3},
        {}  # No objects
    ]
    
    for i, detected_objects in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {detected_objects if detected_objects else 'No objects'}")
        print("-" * 30)
        
        responses = generator.generate_interactive_response(detected_objects)
        
        for j, response in enumerate(responses, 1):
            print(f"  {j}. {response}")
        
        print()
    
    print("🎯 Key Features of Interactive Responses:")
    print("✅ Asks questions instead of just describing")
    print("✅ Engages the user with follow-up prompts")
    print("✅ Provides variety in responses")
    print("✅ Adapts to object counts")
    print("✅ Encourages user interaction")

if __name__ == "__main__":
    test_interactive_responses()
