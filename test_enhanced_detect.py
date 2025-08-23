#!/usr/bin/env python3
"""
Test script for enhanced YOLOv5 detection with AI-powered descriptions and spacebar triggering.
This script demonstrates the new features without requiring a webcam or video stream.
"""

import os
import time
from collections import defaultdict

# Set your OpenAI API key here or as an environment variable
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

def test_ai_descriptions():
    """Test the AI description generation functionality."""
    print("Testing AI Description Generation...")
    
    # Simulate detected objects
    detected_objects = defaultdict(int)
    detected_objects['person'] = 2
    detected_objects['car'] = 1
    detected_objects['laptop'] = 1
    
    print(f"Detected objects: {dict(detected_objects)}")
    
    # Test with AI enabled
    print("\n1. Testing with AI enabled:")
    from detect import TTSManager
    
    # Initialize TTS manager with AI
    tts_manager = TTSManager(
        announcement_interval=2.0,
        ai_api_key=os.getenv('OPENAI_API_KEY'),
        enable_ai=True
    )
    
    # Simulate spacebar press
    tts_manager.spacebar_pressed = True
    tts_manager.update_detections(list(detected_objects.keys()), time.time())
    
    # Wait a bit for TTS to complete
    time.sleep(3)
    
    # Test without AI
    print("\n2. Testing without AI (fallback descriptions):")
    tts_manager_no_ai = TTSManager(
        announcement_interval=2.0,
        enable_ai=False
    )
    
    tts_manager_no_ai.spacebar_pressed = True
    tts_manager_no_ai.update_detections(list(detected_objects.keys()), time.time())
    
    # Wait a bit for TTS to complete
    time.sleep(3)
    
    print("\nTest completed!")

def test_keyboard_monitoring():
    """Test the keyboard monitoring functionality."""
    print("\nTesting Keyboard Monitoring...")
    print("Press SPACEBAR to trigger AI descriptions...")
    print("Press 'q' to quit...")
    
    from detect import TTSManager
    
    # Initialize TTS manager
    tts_manager = TTSManager(
        announcement_interval=5.0,
        ai_api_key=os.getenv('OPENAI_API_KEY'),
        enable_ai=True
    )
    
    # Simulate some detections
    detected_objects = ['person', 'car', 'laptop']
    
    try:
        while True:
            # Simulate detection updates
            current_time = time.time()
            tts_manager.update_detections(detected_objects, current_time)
            
            # Check for quit command
            if input("Press Enter to continue, 'q' to quit: ").lower() == 'q':
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    print("Keyboard monitoring test completed!")

if __name__ == "__main__":
    print("Enhanced YOLOv5 Detection Test")
    print("=" * 40)
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API key found: {api_key[:8]}...")
    else:
        print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable for AI descriptions.")
        print("   The system will use fallback descriptions instead.")
    
    print("\nChoose test mode:")
    print("1. Test AI descriptions")
    print("2. Test keyboard monitoring")
    print("3. Run both tests")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        test_ai_descriptions()
    elif choice == "2":
        test_keyboard_monitoring()
    elif choice == "3":
        test_ai_descriptions()
        test_keyboard_monitoring()
    else:
        print("Invalid choice. Running AI descriptions test...")
        test_ai_descriptions()
