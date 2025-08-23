# YOLOv8 Object Detection with Text-to-Speech

This is a clean, upgraded version of YOLO object detection using the latest YOLOv8 model with integrated text-to-speech functionality.

## Features

- **Latest YOLOv8 Model**: Uses the most recent YOLO architecture for improved accuracy and speed
- **Text-to-Speech Integration**: Announces detected objects in real-time
- **Webcam Support**: Real-time detection with live video feed
- **Image/Video Processing**: Process single images, videos, or directories
- **Customizable TTS**: Adjustable announcement intervals

## Installation

1. **Activate the conda environment:**
   ```bash
   conda activate yolov5
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TTS library:**
   ```bash
   pip install pyttsx3
   ```

## Usage

### Basic Detection with TTS
```bash
# Detect objects in an image with TTS
python detect.py --source data/images/bus.jpg --tts

# Custom TTS interval (default: 5 seconds)
python detect.py --source data/images/bus.jpg --tts --tts-interval 2.0
```

### Webcam Detection
```bash
# Real-time webcam detection with TTS
python detect.py --source 0 --tts --tts-interval 2.0

# Press 'q' to quit webcam mode
```

### Video Processing
```bash
# Process video files
python detect.py --source video.mp4 --tts --tts-interval 3.0
```

### Advanced Options
```bash
# Adjust confidence threshold
python detect.py --source data/images/bus.jpg --tts --conf-thres 0.5

# Save results
python detect.py --source data/images/bus.jpg --tts --save-txt --save-conf
```

## TTS Features

- **Time-based announcements**: Prevents TTS stuttering by announcing at fixed intervals
- **Accurate counting**: Correctly announces multiple instances (e.g., "4 persons" not "1 person")
- **Non-blocking**: TTS runs in background threads, doesn't slow down detection
- **Customizable intervals**: Set announcement frequency with `--tts-interval`

## Model

- **YOLOv8n**: Latest YOLO model for optimal performance
- **COCO Dataset**: 80+ object classes
- **Fast Inference**: Optimized for real-time applications

## Requirements

- Python 3.9+
- PyTorch
- OpenCV
- pyttsx3 (for TTS)
- ultralytics (YOLOv8)

## Directory Structure

```
yolov5/
├── detect.py          # Main detection script (YOLOv8)
├── requirements.txt   # Python dependencies
├── yolov8n.pt        # YOLOv8 model weights
├── data/             # Sample images and datasets
└── runs/             # Detection results
```

## Troubleshooting

### TTS Not Working
- Ensure `pyttsx3` is installed: `pip install pyttsx3`
- Check system audio settings
- On Windows, may need additional TTS drivers

### Webcam Issues
- Ensure camera permissions are granted
- Try different camera indices (0, 1, 2...)
- Check if camera is being used by another application

### Performance Issues
- Lower confidence threshold: `--conf-thres 0.3`
- Use smaller model variants
- Ensure GPU drivers are up to date

## Examples

### Sample Output
```
Text-to-speech engine initialized successfully (announcement interval: 2.0s)
TTS Announcement: Detected: 1 bus, 4 persons, 1 stop sign
TTS announcement completed: Detected: 1 bus, 4 persons, 1 stop sign
```

## License

This project uses the AGPL-3.0 license. See LICENSE file for details.
