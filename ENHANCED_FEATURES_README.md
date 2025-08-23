# Enhanced YOLOv5 Detection with AI-Powered Descriptions

This enhanced version of YOLOv5 detection includes AI-powered object descriptions and spacebar-triggered voice announcements.

## 🚀 New Features

### 1. AI-Powered Object Descriptions
- **OpenAI Integration**: Uses GPT models to generate engaging, contextual descriptions of detected objects
- **Smart Fallbacks**: Includes built-in fallback descriptions when AI is unavailable
- **Customizable**: Supports different AI models and API configurations

### 2. Spacebar-Triggered Voice Announcements
- **Instant Activation**: Press SPACEBAR to trigger AI-powered descriptions immediately
- **Non-blocking**: Voice announcements run in background threads
- **Smart Timing**: Prevents duplicate announcements and manages intervals

### 3. Enhanced Text-to-Speech
- **Multiple Voices**: Configurable speech rate and volume
- **Thread-Safe**: Handles concurrent detection and speech operations
- **Error Handling**: Graceful fallbacks when TTS fails

## 📋 Requirements

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Additional Dependencies
```bash
pip install openai keyboard pyttsx3
```

### Optional: OpenAI API Key
For AI-powered descriptions, you'll need an OpenAI API key:
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set it as an environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY=your-api-key-here
   ```

## 🎯 Usage Examples

### Basic Detection with AI Descriptions
```bash
# Enable TTS and AI descriptions
python detect.py --weights yolov5s.pt --source 0 --tts --enable-ai

# With custom API key
python detect.py --weights yolov5s.pt --source 0 --tts --enable-ai --ai-api-key your-key-here
```

### Image Detection with Enhanced Features
```bash
# Process images with AI descriptions
python detect.py --weights yolov5s.pt --source data/images/bus.jpg --tts --enable-ai --view-img
```

### Video Processing with Custom TTS Interval
```bash
# Process video with 3-second TTS intervals
python detect.py --weights yolov5s.pt --source video.mp4 --tts --tts-interval 3.0 --enable-ai
```

## ⌨️ Spacebar Controls

### How It Works
1. **Automatic Mode**: Objects are announced every `--tts-interval` seconds
2. **Manual Mode**: Press SPACEBAR to trigger immediate AI-powered descriptions
3. **Hybrid Mode**: Both automatic and manual triggers work together

### Spacebar Behavior
- **Single Press**: Triggers AI description generation for current detections
- **Multiple Presses**: Each press generates new descriptions
- **No Interruption**: Previous announcements complete before new ones start

## 🤖 AI Description Examples

### With OpenAI (GPT-3.5/4)
```
"AI Analysis: 1 person: A person in casual attire standing confidently. 1 car: A sleek modern vehicle with aerodynamic design. 1 laptop: A portable computing device ready for work."
```

### Fallback Descriptions (No AI)
```
"AI Analysis: 1 person: A person in the scene. 1 car: A vehicle on the road. 1 laptop: A portable computing device."
```

## ⚙️ Configuration Options

### Command Line Arguments
```bash
--tts                    # Enable text-to-speech
--tts-interval 5.0      # TTS announcement interval (seconds)
--enable-ai             # Enable AI-powered descriptions (default: True)
--ai-api-key KEY        # OpenAI API key
```

### Environment Variables
```bash
OPENAI_API_KEY=your-key-here    # OpenAI API key
```

## 🔧 Customization

### Modifying Fallback Descriptions
Edit the `_generate_fallback_descriptions` method in `AIDescriptionGenerator` class:

```python
fallback_descriptions = {
    "person": "A human being in the scene",
    "car": "An automobile vehicle",
    # Add your custom descriptions here
}
```

### Changing AI Models
Modify the model parameter in `AIDescriptionGenerator`:

```python
ai_generator = AIDescriptionGenerator(
    api_key=api_key,
    model="gpt-4"  # or "gpt-3.5-turbo"
)
```

### TTS Voice Settings
Adjust speech properties in `TTSManager`:

```python
self.engine.setProperty('rate', 150)      # Speed (words per minute)
self.engine.setProperty('volume', 0.8)    # Volume (0.0 to 1.0)
```

## 🧪 Testing

### Test Script
Run the included test script to verify functionality:

```bash
python test_enhanced_detect.py
```

### Test Modes
1. **AI Descriptions**: Test with and without AI enabled
2. **Keyboard Monitoring**: Test spacebar functionality
3. **Both Tests**: Run comprehensive testing

## 🚨 Troubleshooting

### Common Issues

#### TTS Not Working
```bash
# Check if pyttsx3 is installed
pip install pyttsx3

# On Windows, ensure pywin32 is available
pip install pywin32
```

#### Keyboard Monitoring Fails
```bash
# Check if keyboard package is installed
pip install keyboard

# On some systems, may need root/admin privileges
```

#### AI Descriptions Not Working
```bash
# Verify OpenAI API key
echo $OPENAI_API_KEY

# Check OpenAI package installation
pip install openai

# Test API connectivity
python -c "import openai; print('OpenAI available')"
```

### Performance Tips
- **GPU Acceleration**: Use `--device 0` for CUDA acceleration
- **Batch Processing**: Process multiple images with `--source path/to/images/`
- **Model Selection**: Use smaller models (yolov5n, yolov5s) for faster inference

## 📝 API Reference

### TTSManager Class
```python
class TTSManager:
    def __init__(self, announcement_interval=5.0, ai_api_key=None, enable_ai=True):
        """
        Initialize TTS manager with AI integration.
        
        Args:
            announcement_interval: Time between automatic announcements
            ai_api_key: OpenAI API key for AI descriptions
            enable_ai: Whether to enable AI-powered descriptions
        """
```

### AIDescriptionGenerator Class
```python
class AIDescriptionGenerator:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize AI description generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
    
    def generate_descriptions(self, detected_objects):
        """
        Generate AI-powered descriptions for detected objects.
        
        Args:
            detected_objects: Dictionary of object counts
            
        Returns:
            List of description strings
        """
```

## 🤝 Contributing

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Reporting Issues
- Check existing issues first
- Provide detailed error messages
- Include system information and Python version
- Describe steps to reproduce

## 📄 License

This enhanced version maintains the original AGPL-3.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Ultralytics**: Original YOLOv5 implementation
- **OpenAI**: AI-powered description generation
- **pyttsx3**: Text-to-speech functionality
- **keyboard**: Cross-platform keyboard monitoring

---

**Note**: This enhancement requires an active internet connection for AI-powered descriptions. Offline mode uses fallback descriptions.
