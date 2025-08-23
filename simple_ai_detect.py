#!/usr/bin/env python3
"""
Simple AI-Powered YOLOv5 Detection with Camera Integration
Lightweight version for better performance
"""

import cv2
import torch
import time
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

# Add the current directory to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detect import TTSManager
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

class SimpleAIDescriptionGenerator:
    """Interactive description generator with user engagement features."""
    
    def __init__(self):
        self.interactive_responses = {
            "person": [
                "I can see a person! What would you like to know about them?",
                "There's a human in the frame. Should I analyze their position?",
                "A person detected! Would you like me to describe their location?",
                "Human presence confirmed. What interests you about this person?",
                "I see someone! What would you like me to tell you?"
            ],
            "car": [
                "A vehicle spotted! What would you like to know about it?",
                "Car detected in the scene. Should I analyze its type?",
                "There's a car! What aspects would you like me to focus on?",
                "Vehicle identified! What information do you need?",
                "I can see a car! What would you like me to describe?"
            ],
            "truck": [
                "A large vehicle detected! What would you like to know?",
                "Truck spotted! Should I analyze its size and position?",
                "There's a truck! What interests you about it?",
                "Heavy vehicle identified! What details do you need?",
                "I see a truck! What would you like me to tell you?"
            ],
            "bus": [
                "A bus is in the frame! What would you like to know?",
                "Public transport detected! Should I analyze its route?",
                "There's a bus! What aspects would you like me to focus on?",
                "Bus identified! What information do you need?",
                "I can see a bus! What would you like me to describe?"
            ],
            "motorcycle": [
                "A two-wheeler spotted! What would you like to know?",
                "Motorcycle detected! Should I analyze its type?",
                "There's a bike! What interests you about it?",
                "Two-wheeler identified! What details do you need?",
                "I see a motorcycle! What would you like me to tell you?"
            ],
            "bicycle": [
                "A bicycle detected! What would you like to know?",
                "Bike spotted! Should I analyze its position?",
                "There's a bicycle! What aspects would you like me to focus on?",
                "Bike identified! What information do you need?",
                "I can see a bicycle! What would you like me to describe?"
            ],
            "dog": [
                "A furry friend spotted! What would you like to know?",
                "Dog detected! Should I analyze its behavior?",
                "There's a canine! What interests you about it?",
                "Dog identified! What details do you need?",
                "I see a dog! What would you like me to tell you?"
            ],
            "cat": [
                "A feline friend detected! What would you like to know?",
                "Cat spotted! Should I analyze its position?",
                "There's a cat! What aspects would you like me to focus on?",
                "Cat identified! What information do you need?",
                "I can see a cat! What would you like me to describe?"
            ],
            "laptop": [
                "A computing device spotted! What would you like to know?",
                "Laptop detected! Should I analyze its type?",
                "There's a computer! What interests you about it?",
                "Laptop identified! What details do you need?",
                "I can see a laptop! What would you like me to tell you?"
            ],
            "phone": [
                "A mobile device detected! What would you like to know?",
                "Phone spotted! Should I analyze its position?",
                "There's a phone! What aspects would you like me to focus on?",
                "Phone identified! What information do you need?",
                "I can see a phone! What would you like me to describe?"
            ],
            "chair": [
                "A seating option detected! What would you like to know?",
                "Chair spotted! Should I analyze its comfort level?",
                "There's a chair! What interests you about it?",
                "Chair identified! What details do you need?",
                "I can see a chair! What would you like me to tell you?"
            ],
            "table": [
                "A surface detected! What would you like to know?",
                "Table spotted! Should I analyze its size?",
                "There's a table! What aspects would you like me to focus on?",
                "Table identified! What information do you need?",
                "I can see a table! What would you like me to describe?"
            ]
        }
        
        # Initialize random seed for variety
        import random
        random.seed(time.time())
        
        # Track user interactions
        self.user_interaction_count = 0
        self.last_interaction_time = 0
    
    def generate_interactive_response(self, detected_objects):
        """Generate interactive responses for detected objects."""
        import random
        
        if not detected_objects:
            return ["No objects detected. What would you like me to look for?"]
        
        responses = []
        for obj_name, count in detected_objects.items():
            if obj_name in self.interactive_responses:
                # Pick a random interactive response
                response = random.choice(self.interactive_responses[obj_name])
                
                # Add count information
                if count > 1:
                    response = f"I found {count} {obj_name}s! {response}"
                else:
                    response = f"I found 1 {obj_name}! {response}"
                
                responses.append(response)
            else:
                # Generic interactive response for unknown objects
                if count > 1:
                    response = f"I detected {count} {obj_name}s! What would you like to know about them?"
                else:
                    response = f"I detected 1 {obj_name}! What would you like to know about it?"
                responses.append(response)
        
        # Add engagement prompts
        if len(responses) > 1:
            responses.append("Multiple objects detected! Which one interests you most?")
        else:
            responses.append("What would you like me to focus on?")
        
        return responses
    
    def generate_descriptions(self, detected_objects):
        """Generate interactive responses instead of static descriptions."""
        return self.generate_interactive_response(detected_objects)


class CameraDetector:
    """Real-time camera detection with AI-powered descriptions."""
    
    def __init__(self, weights="yolov5s.pt", camera_id=0):
        self.weights = weights
        self.camera_id = camera_id
        self.device = select_device("")
        self.model = None
        self.names = None
        self.stride = None
        self.imgsz = (640, 640)
        self.cap = None
        self.running = False
        
        # Initialize AI description generator
        self.ai_generator = SimpleAIDescriptionGenerator()
        
        # Initialize TTS manager with custom AI generator
        self.tts_manager = TTSManager(
            announcement_interval=3.0,
            enable_ai=True
        )
        
        # Track spacebar presses for visual feedback
        self.last_spacebar_time = 0
        
        # Track current detections for spacebar responses
        self.current_detections = []
        
        # Override the AI generator in TTS manager
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
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
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
            annotator = Annotator(frame, line_width=2, example=str(self.names))
            
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
            
            # Store current detections for spacebar responses
            self.current_detections = detected_objects.copy()
            
            return annotator.result()
        
        return frame
    
    def run_detection(self):
        """Main detection loop."""
        if not self.start_camera():
            return
        
        print("🚀 Starting real-time interactive detection...")
        print("📱 Press SPACEBAR for interactive responses")
        print("📱 Press 'q' to quit")
        print("📱 Press 's' to save current frame")
        print("📱 Press 'r' to reset TTS timer")
        print("💬 The system will ask you questions about detected objects!")
        
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
                
                # Add info text to frame
                cv2.putText(annotated_frame, "Press SPACE for interactive responses", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add spacebar status indicator
                if hasattr(self, 'last_spacebar_time') and time.time() - self.last_spacebar_time < 2.0:
                    cv2.putText(annotated_frame, "SPACEBAR ACTIVATED!", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
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
                    # Trigger interactive response manually
                    if self.tts_manager:
                        # Update spacebar timestamp for visual feedback
                        self.last_spacebar_time = time.time()
                        
                        # Force the TTS manager to generate interactive responses
                        current_time = time.time()
                        detected_objects_dict = defaultdict(int)
                        for obj in self.current_detections:
                            detected_objects_dict[obj] += 1
                        
                        # Generate and speak interactive responses
                        responses = self.ai_generator.generate_interactive_response(detected_objects_dict)
                        print(f"🎯 Interactive responses generated:")
                        for i, response in enumerate(responses, 1):
                            print(f"  {i}. {response}")
                        
                        # Print the interactive responses clearly
                        print(f"🎯 Interactive responses generated:")
                        for i, response in enumerate(responses, 1):
                            print(f"  {i}. {response}")
                        
                        # Use a simple TTS approach to avoid conflicts
                        if self.tts_manager and self.tts_manager.engine:
                            try:
                                # Create a simple announcement
                                simple_announcement = f"I found {len(detected_objects_dict)} different objects. {responses[0] if responses else 'What would you like to know?'}"
                                
                                # Use the engine directly with a new thread to avoid conflicts
                                def speak_response():
                                    try:
                                        self.tts_manager.engine.say(simple_announcement)
                                        self.tts_manager.engine.runAndWait()
                                    except Exception as e:
                                        print(f"TTS Error: {e}")
                                
                                # Run TTS in a separate thread
                                import threading
                                tts_thread = threading.Thread(target=speak_response)
                                tts_thread.daemon = True
                                tts_thread.start()
                                
                            except Exception as e:
                                print(f"🔊 TTS Error: {e}")
                                print("💬 Interactive responses (text only):")
                                for response in responses[:3]:
                                    print(f"   {response}")
                        else:
                            print("💬 Interactive responses (text only):")
                            for response in responses[:3]:
                                print(f"   {response}")
                        
                        print("🔊 Spacebar pressed - triggering interactive response!")
                        print("💬 I'm ready to engage with you about the detected objects!")
                        print(f"📝 Generated {len(responses)} interactive responses")
                elif key == ord('r'):
                    # Reset TTS timer
                    if self.tts_manager:
                        self.tts_manager.last_announcement_time = 0
                        print("🔄 TTS timer reset")
                
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
    """Main function to run the interactive camera detection."""
    print("🤖 Interactive YOLOv5 Camera Detection")
    print("=" * 50)
    
    # Check if camera is available
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ No camera found. Please check your camera connection.")
        print("💡 Try different camera IDs (0, 1, 2) if you have multiple cameras")
        return
    test_cap.release()
    
    # Check if YOLOv5 weights exist
    weights_file = "yolov5s.pt"
    if not Path(weights_file).exists():
        print(f"❌ YOLOv5 weights file '{weights_file}' not found.")
        print("💡 The script will download it automatically on first run.")
    
    # Initialize detector
    try:
        detector = CameraDetector(
            weights=weights_file,
            camera_id=0
        )
        
        # Run detection
        detector.run_detection()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the required dependencies installed")


if __name__ == "__main__":
    main()
