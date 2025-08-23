#!/usr/bin/env python3
"""
YOLOv8 Object Detection with Text-to-Speech Integration
=======================================================

This script provides object detection using the latest YOLOv8 model with integrated
text-to-speech announcements for detected objects. It maintains the same TTS functionality
as the original YOLOv5 implementation while using the modern Ultralytics YOLOv8 API.

Usage:
    $ python detect_v8.py --model yolov8n.pt --source 0                    # webcam
    $ python detect_v8.py --model yolov8n.pt --source img.jpg             # image
    $ python detect_v8.py --model yolov8n.pt --source vid.mp4             # video
    $ python detect_v8.py --model yolov8n.pt --source path/               # directory
    $ python detect_v8.py --model yolov8n.pt --source 'path/*.jpg'        # glob
    $ python detect_v8.py --model yolov8n.pt --source 'https://youtu.be/LNwODJXcvt4'  # YouTube
    $ python detect_v8.py --model yolov8n.pt --source 'rtsp://example.com/media.mp4'  # RTSP stream

With TTS:
    $ python detect_v8.py --model yolov8n.pt --source 0 --tts              # webcam with TTS
    $ python detect_v8.py --model yolov8n.pt --source vid.mp4 --tts --tts-interval 3.0  # video with TTS every 3s
"""

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path
from collections import defaultdict
import threading
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Try to import pyttsx3 for text-to-speech functionality
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")


class TTSManager:
    """Manages text-to-speech functionality with time-based announcements."""
    
    def __init__(self, announcement_interval: float = 5.0):
        self.engine = None
        self.lock = threading.Lock()
        self.last_announcement = ""
        self.last_announcement_time = 0
        self.announcement_interval = announcement_interval
        self.current_detections = defaultdict(int)  # Track object counts
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                # Configure TTS settings
                self.engine.setProperty('rate', 150)  # Speed of speech
                self.engine.setProperty('volume', 0.8)  # Volume level
                print(f"Text-to-speech engine initialized successfully (announcement interval: {announcement_interval}s)")
            except Exception as e:
                print(f"Warning: Failed to initialize text-to-speech engine: {e}")
                self.engine = None
        else:
            print("Warning: pyttsx3 not available. TTS functionality disabled.")
    
    def update_detections(self, detections: List[str], current_time: float):
        """Update current detections and announce if interval has passed."""
        if not self.engine:
            return
            
        # Convert detections list to count dictionary for accurate comparison
        new_detections = defaultdict(int)
        for detection in detections:
            new_detections[detection] += 1
        
        # Check if it's time to announce and if detections have changed
        time_since_last = current_time - self.last_announcement_time
        
        # Announce if: interval has passed AND detections changed, OR this is the first detection
        should_announce = (
            (time_since_last >= self.announcement_interval and new_detections != self.current_detections) or
            (not self.current_detections and new_detections)  # First detection
        )
        
        if should_announce:
            self.current_detections = new_detections.copy()
            self.last_announcement_time = current_time
            
            if new_detections:
                self._announce_detections(new_detections)
            else:
                print("TTS: No objects detected")
    
    def _announce_detections(self, detections: Dict[str, int]):
        """Announce detected objects in a non-blocking way."""
        announcement_parts = []
        for obj_name, count in detections.items():
            if count == 1:
                announcement_parts.append(f"1 {obj_name}")
            else:
                announcement_parts.append(f"{count} {obj_name}s")
        
        announcement = f"Detected: {', '.join(announcement_parts)}"
        
        # Log the announcement for debugging
        print(f"TTS Announcement: {announcement}")
        
        # Use threading to avoid blocking the main detection loop
        def speak():
            with self.lock:
                try:
                    self.engine.say(announcement)
                    self.engine.runAndWait()
                    print(f"TTS announcement completed: {announcement}")
                except Exception as e:
                    print(f"Warning: TTS announcement failed: {e}")
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()

class YOLOv8Detector:
    """YOLOv8-based object detector with TTS integration."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tts_manager = None
        
    def set_tts(self, enabled: bool, interval: float = 5.0):
        """Enable or disable TTS functionality."""
        if enabled:
            if not TTS_AVAILABLE:
                print("Warning: TTS requested but pyttsx3 is not available. Install with: pip install pyttsx3")
                return
            self.tts_manager = TTSManager(announcement_interval=interval)
            print(f"TTS enabled with {interval}s announcement interval")
        else:
            self.tts_manager = None
    
    def detect_image(self, image_path: str, save_path: Optional[str] = None, 
                    show_result: bool = False) -> List[Dict]:
        """Detect objects in a single image."""
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'class': int(box.cls[0]),
                        'class_name': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    detections.append(detection)
        
        # Process TTS if enabled
        if self.tts_manager:
            detected_objects = [det['class_name'] for det in detections]
            current_time = time.time()
            self.tts_manager.update_detections(detected_objects, current_time)
        
        # Save or show result
        if save_path or show_result:
            annotated_img = results[0].plot()
            if save_path:
                cv2.imwrite(save_path, annotated_img)
            if show_result:
                cv2.imshow('YOLOv8 Detection', annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return detections
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None,
                    show_result: bool = False, save_csv: bool = False) -> List[Dict]:
        """Detect objects in a video file or stream."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup CSV writer if requested
        csv_writer = None
        csv_file = None
        if save_csv:
            csv_path = output_path.replace('.mp4', '.csv') if output_path else 'detections.csv'
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame', 'Class', 'Class_Name', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        
        frame_count = 0
        all_detections = []
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                
                frame_detections = []
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            detection = {
                                'frame': frame_count,
                                'class': int(box.cls[0]),
                                'class_name': result.names[int(box.cls[0])],
                                'confidence': float(box.conf[0]),
                                'bbox': box.xyxy[0].cpu().numpy().tolist()
                            }
                            frame_detections.append(detection)
                            
                            # Write to CSV if requested
                            if csv_writer:
                                x1, y1, x2, y2 = detection['bbox']
                                csv_writer.writerow([
                                    frame_count, detection['class'], detection['class_name'],
                                    detection['confidence'], x1, y1, x2, y2
                                ])
                
                all_detections.extend(frame_detections)
                
                # Process TTS if enabled
                if self.tts_manager:
                    detected_objects = [det['class_name'] for det in frame_detections]
                    current_time = time.time() - start_time
                    self.tts_manager.update_detections(detected_objects, current_time)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame if requested
                if show_result:
                    cv2.imshow('YOLOv8 Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"Processed {frame_count} frames...")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if show_result:
                cv2.destroyAllWindows()
        
        print(f"Video processing completed. Total frames: {frame_count}")
        return all_detections
    
    def detect_webcam(self, camera_id: int = 0, show_result: bool = True):
        """Detect objects from webcam feed."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print(f"Starting webcam detection on camera {camera_id}. Press 'q' to quit.")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                
                # Process TTS if enabled
                if self.tts_manager:
                    detected_objects = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                class_name = result.names[int(box.cls[0])]
                                detected_objects.append(class_name)
                    
                    current_time = time.time() - start_time
                    self.tts_manager.update_detections(detected_objects, current_time)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Show frame
                if show_result:
                    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam detection stopped.")


def main():
    """Main function to run YOLOv8 detection with TTS."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection with TTS')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image, video, directory, or camera ID (0 for webcam)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--show', action='store_true',
                       help='Show results in window')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save detection results to CSV')
    parser.add_argument('--tts', action='store_true',
                       help='Enable text-to-speech announcements')
    parser.add_argument('--tts-interval', type=float, default=5.0,
                       help='TTS announcement interval in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv8Detector(args.model, args.conf, args.iou)
    
    # Setup TTS if requested
    if args.tts:
        detector.set_tts(True, args.tts_interval)
    
    # Determine source type and run detection
    source = args.source
    
    if source.isdigit() or source == '0':
        # Webcam
        camera_id = int(source)
        detector.detect_webcam(camera_id, args.show)
    
    elif os.path.isfile(source):
        # Single file (image or video)
        file_ext = Path(source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Image
            print(f"Detecting objects in image: {source}")
            detections = detector.detect_image(source, args.output, args.show)
            print(f"Found {len(detections)} objects")
            for det in detections:
                print(f"  {det['class_name']}: {det['confidence']:.2f}")
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            print(f"Detecting objects in video: {source}")
            detections = detector.detect_video(source, args.output, args.show, args.save_csv)
            print(f"Total detections: {len(detections)}")
    
    elif os.path.isdir(source):
        # Directory of images
        print(f"Processing directory: {source}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(source).glob(f'*{ext}'))
            image_files.extend(Path(source).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in directory: {source}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            detections = detector.detect_image(str(img_path), None, False)
            print(f"  Found {len(detections)} objects")
    
    else:
        print(f"Error: Invalid source '{source}'. Please provide a valid file, directory, or camera ID.")
        return


if __name__ == "__main__":
    main()

