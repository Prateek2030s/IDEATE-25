#!/usr/bin/env python3
"""
Free AI-Powered YOLOv5 Detection with Camera Integration
Uses Hugging Face models for free AI descriptions and integrates with webcam
"""

import cv2
import torch
import time
import threading
from collections import defaultdict
import numpy as np
from pathlib import Path
import sys

# Add the current directory to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detect import TTSManager, AIDescriptionGenerator
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

class FreeAIDescriptionGenerator:
    """Free AI description generator using Hugging Face models."""
    
    def __init__(self, model_name="gpt2"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            # Use a smaller, faster model for real-time processing
            if self.model_name == "gpt2":
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            else:
                # Fallback to a simple text generation pipeline
                self.model = pipeline("text-generation", model="gpt2", max_length=50)
                
            print(f"✅ Free AI model loaded: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Failed to load AI model: {e}")
            print("   Using fallback descriptions instead.")
            self.model = None
    
    def generate_descriptions(self, detected_objects):
        """Generate AI-powered descriptions for detected objects."""
        if not self.model or not detected_objects:
            return self._generate_fallback_descriptions(detected_objects)
        
        try:
            descriptions = []
            for obj_name, count in detected_objects.items():
                # Create a simple prompt for the model
                prompt = f"I see a {obj_name} in the scene. It looks like"
                
                if hasattr(self.model, 'generate'):
                    # For pipeline models
                    inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                    outputs = self.model.generate(
                        inputs, 
                        max_length=len(inputs[0]) + 15,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract the generated part
                    description = generated_text[len(prompt):].strip()
                    if description:
                        descriptions.append(f"{description}")
                    else:
                        descriptions.append(self._get_fallback_description(obj_name))
                else:
                    # For pipeline models
                    result = self.model(prompt, max_length=len(prompt.split()) + 10)
                    generated_text = result[0]['generated_text']
                    description = generated_text[len(prompt):].strip()
                    if description:
                        descriptions.append(f"{description}")
                    else:
                        descriptions.append(self._get_fallback_description(obj_name))
            
            return descriptions[:len(detected_objects)]
            
        except Exception as e:
            print(f"⚠️  AI description generation failed: {e}")
            return self._generate_fallback_descriptions(detected_objects)
    
    def _get_fallback_description(self, obj_name):
        """Get a fallback description for a specific object."""
        fallback_descriptions = {
            "person": "a human being in the scene",
            "car": "a vehicle on the road",
            "truck": "a large transport vehicle",
            "bus": "a public transport bus",
            "motorcycle": "a two-wheeled vehicle",
            "bicycle": "a pedal-powered bicycle",
            "dog": "a friendly canine companion",
            "cat": "a feline friend",
            "bird": "a feathered creature",
            "chair": "a seating furniture piece",
            "table": "a flat surface for work or dining",
            "laptop": "a portable computing device",
            "phone": "a mobile communication device",
            "book": "a source of knowledge and stories",
            "cup": "a container for beverages",
            "bottle": "a vessel for liquids",
            "bag": "a container for carrying items",
            "backpack": "a bag worn on the back",
            "umbrella": "protection from rain or sun",
            "clock": "a time-keeping device"
        }
        
        return fallback_descriptions.get(obj_name, f"a {obj_name} in the scene")
    
    def _generate_fallback_descriptions(self, detected_objects):
        """Generate simple fallback descriptions when AI is not available."""
        descriptions = []
        for obj in detected_objects.keys():
            descriptions.append(self._get_fallback_description(obj))
        return descriptions


class CameraAIDetector:
    """Real-time camera detection with AI-powered descriptions."""
    
    def __init__(self, weights="yolov5s.pt", camera_id=0, enable_ai=True):
        self.weights = weights
        self.camera_id = camera_id
        self.enable_ai = enable_ai
        self.device = select_device("")
        self.model = None
        self.names = None
        self.stride = None
        self.imgsz = (640, 640)
        self.cap = None
        self.running = False
        
        # Initialize AI description generator
        if enable_ai:
            self.ai_generator = FreeAIDescriptionGenerator()
        else:
            self.ai_generator = None
        
        # Initialize TTS manager
        self.tts_manager = TTSManager(
            announcement_interval=3.0,
            enable_ai=enable_ai
        )
        
        # Override the AI generator in TTS manager
        if enable_ai:
            self.tts_manager.ai_generator = self.ai_generator
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv5 model."""
        try:
            self.model = DetectMultiBackend(self.weights, device=self.device)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            
            # Warm up the model
            self.model.warmup(imgsz=(1, 3, *self.imgsz))
            print(f"✅ YOLOv5 model loaded: {self.weights}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def start_camera(self):
        """Start the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {self.camera_id}")
                return False
            
            print(f"✅ Camera {self.camera_id} started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for object detection."""
        # Preprocess the frame
        im = cv2.resize(frame, self.imgsz)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        # Inference
        pred = self.model(im)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
        
        return pred, im
    
    def draw_detections(self, frame, pred, im):
        """Draw detection boxes on the frame."""
        if len(pred[0]):
            # Rescale boxes from img_size to frame size
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
            
            # Create annotator
            annotator = Annotator(frame, line_width=3, example=str(self.names))
            
            # Collect detected objects for TTS
            detected_objects = []
            
            # Process each detection
            for *xyxy, conf, cls in pred[0]:
                c = int(cls)
                label = f"{self.names[c]} {conf:.2f}"
                detected_objects.append(self.names[c])
                
                # Draw bounding box
                annotator.box_label(xyxy, label, color=colors(c, True))
            
            # Update TTS with current detections
            if self.tts_manager:
                current_time = time.time()
                self.tts_manager.update_detections(detected_objects, current_time)
            
            return annotator.result()
        
        return frame
    
    def run_detection(self):
        """Main detection loop."""
        if not self.start_camera():
            return
        
        print("🚀 Starting real-time detection...")
        print("📱 Press SPACEBAR for AI descriptions")
        print("📱 Press 'q' to quit")
        print("📱 Press 's' to save current frame")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                pred, im = self.process_frame(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, pred, im)
                
                # Display frame
                cv2.imshow("AI-Powered Object Detection", annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord(' '):
                    # Trigger AI description manually
                    if self.tts_manager:
                        self.tts_manager.spacebar_pressed = True
                        print("🔊 Spacebar pressed - triggering AI description")
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"📊 FPS: {fps:.1f}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Detection interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detection and clean up."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Detection stopped")


def main():
    """Main function to run the AI-powered camera detection."""
    print("🤖 Free AI-Powered YOLOv5 Camera Detection")
    print("=" * 50)
    
    # Check if camera is available
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ No camera found. Please check your camera connection.")
        return
    test_cap.release()
    
    # Initialize detector
    try:
        detector = CameraAIDetector(
            weights="yolov5s.pt",
            camera_id=0,
            enable_ai=True
        )
        
        # Run detection
        detector.run_detection()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the YOLOv5 weights file (yolov5s.pt) in the current directory")


if __name__ == "__main__":
    main()
