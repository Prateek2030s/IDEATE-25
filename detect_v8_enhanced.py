#!/usr/bin/env python3
"""
Enhanced YOLOv8 Object Detection with Text-to-Speech Integration
================================================================

This enhanced script provides object detection using the latest YOLOv8 model with integrated
text-to-speech announcements. It maintains full compatibility with the original TTS functionality
while using the modern Ultralytics YOLOv8 API and configuration-based settings.

Features:
- YOLOv8 model support (latest stable release)
- Configurable TTS with time-based announcements
- Support for images, videos, webcam, and directories
- CSV and text output formats
- Performance optimization options
- Configuration file support

Usage:
    $ python detect_v8_enhanced.py --config config_v8.yaml --source 0
    $ python detect_v8_enhanced.py --model yolov8n.pt --source img.jpg --tts
    $ python detect_v8_enhanced.py --source vid.mp4 --tts --tts-interval 3.0
"""

import argparse
import csv
import logging
import os
import platform
import sys
import time
from pathlib import Path
from collections import defaultdict
import threading
from typing import List, Dict, Optional, Tuple, Union
import yaml

import cv2
import numpy as np
import torch
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
    """Enhanced text-to-speech manager with configurable settings."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.engine = None
        self.lock = threading.Lock()
        self.last_announcement = ""
        self.last_announcement_time = 0
        self.announcement_interval = config.get('announcement_interval', 5.0)
        self.current_detections = defaultdict(int)
        
        if TTS_AVAILABLE and config.get('enabled', False):
            try:
                self.engine = pyttsx3.init()
                
                # Configure TTS settings from config
                self.engine.setProperty('rate', config.get('speech_rate', 150))
                self.engine.setProperty('volume', config.get('volume', 0.8))
                
                # Set specific voice if provided
                voice_id = config.get('voice_id')
                if voice_id:
                    voices = self.engine.getProperty('voices')
                    if voice_id < len(voices):
                        self.engine.setProperty('voice', voices[voice_id].id)
                
                logging.info(f"TTS engine initialized: rate={config.get('speech_rate', 150)}, "
                           f"volume={config.get('volume', 0.8)}, interval={self.announcement_interval}s")
            except Exception as e:
                logging.warning(f"Failed to initialize TTS engine: {e}")
                self.engine = None
        else:
            if not TTS_AVAILABLE:
                logging.warning("pyttsx3 not available. TTS functionality disabled.")
            else:
                logging.info("TTS disabled in configuration.")
    
    def update_detections(self, detections: List[str], current_time: float):
        """Update current detections and announce if interval has passed."""
        if not self.engine:
            return
            
        # Convert detections list to count dictionary
        new_detections = defaultdict(int)
        for detection in detections:
            new_detections[detection] += 1
        
        # Check if it's time to announce and if detections have changed
        time_since_last = current_time - self.last_announcement_time
        
        should_announce = (
            (time_since_last >= self.announcement_interval and new_detections != self.current_detections) or
            (not self.current_detections and new_detections)
        )
        
        if should_announce:
            self.current_detections = new_detections.copy()
            self.last_announcement_time = current_time
            
            if new_detections:
                self._announce_detections(new_detections)
            else:
                logging.info("TTS: No objects detected")
    
    def _announce_detections(self, detections: Dict[str, int]):
        """Announce detected objects in a non-blocking way."""
        announcement_parts = []
        for obj_name, count in detections.items():
            if count == 1:
                announcement_parts.append(f"1 {obj_name}")
            else:
                announcement_parts.append(f"{count} {obj_name}s")
        
        announcement = f"Detected: {', '.join(announcement_parts)}"
        logging.info(f"TTS Announcement: {announcement}")
        
        def speak():
            with self.lock:
                try:
                    self.engine.say(announcement)
                    self.engine.runAndWait()
                    logging.info(f"TTS announcement completed: {announcement}")
                except Exception as e:
                    logging.warning(f"TTS announcement failed: {e}")
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()


class YOLOv8Detector:
    """Enhanced YOLOv8-based object detector with TTS integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tts_manager = None
        self._load_model()
        self._setup_tts()
        
    def _load_model(self):
        """Load YOLOv8 model with configuration."""
        model_path = self.config['model']['path']
        
        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded YOLOv8 model: {model_path}")
            
            # Set device
            device = self.config['performance']['device']
            if device != "auto":
                self.model.to(device)
                logging.info(f"Model moved to device: {device}")
                
        except Exception as e:
            logging.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def _setup_tts(self):
        """Setup TTS manager if enabled."""
        if self.config['tts']['enabled']:
            self.tts_manager = TTSManager(self.config['tts'])
    
    def detect_image(self, image_path: str, save_path: Optional[str] = None, 
                    show_result: bool = False) -> List[Dict]:
        """Detect objects in a single image."""
        results = self.model(
            image_path, 
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['iou_threshold'],
            max_det=self.config['model']['max_detections'],
            classes=self.config['filtering']['classes'],
            agnostic_nms=self.config['filtering']['agnostic_nms'],
            augment=self.config['filtering']['augment']
        )
        
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
            annotated_img = results[0].plot(
                line_width=self.config['display']['line_thickness'],
                labels=not self.config['display']['hide_labels'],
                conf=not self.config['display']['hide_confidence']
            )
            if save_path:
                cv2.imwrite(save_path, annotated_img)
                logging.info(f"Saved annotated image: {save_path}")
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
            logging.error(f"Could not open video source: {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*self.config['output']['video_codec'])
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
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['iou_threshold'],
                    max_det=self.config['model']['max_detections'],
                    classes=self.config['filtering']['classes'],
                    agnostic_nms=self.config['filtering']['agnostic_nms'],
                    augment=self.config['filtering']['augment']
                )
                
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
                annotated_frame = results[0].plot(
                    line_width=self.config['display']['line_thickness'],
                    labels=not self.config['display']['hide_labels'],
                    conf=not self.config['display']['hide_confidence']
                )
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame if requested
                if show_result:
                    cv2.imshow('YOLOv8 Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logging.info(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if show_result:
                cv2.destroyAllWindows()
        
        logging.info(f"Video processing completed. Total frames: {frame_count}")
        return all_detections
    
    def detect_webcam(self, camera_id: int = 0, show_result: bool = True):
        """Detect objects from webcam feed."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Could not open camera {camera_id}")
            return
        
        logging.info(f"Starting webcam detection on camera {camera_id}. Press 'q' to quit.")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Could not read frame from camera")
                    break
                
                # Run detection
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['iou_threshold'],
                    max_det=self.config['model']['max_detections'],
                    classes=self.config['filtering']['classes'],
                    agnostic_nms=self.config['filtering']['agnostic_nms'],
                    augment=self.config['filtering']['augment']
                )
                
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
                annotated_frame = results[0].plot(
                    line_width=self.config['display']['line_thickness'],
                    labels=not self.config['display']['hide_labels'],
                    conf=not self.config['display']['hide_confidence']
                )
                
                # Show frame
                if show_result:
                    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Webcam detection stopped.")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def setup_logging(config: Dict):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['log_level'].upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yolov8_detection.log') if config['logging']['save_logs'] else logging.NullHandler()
        ]
    )


def main():
    """Main function to run enhanced YOLOv8 detection with TTS."""
    parser = argparse.ArgumentParser(description='Enhanced YOLOv8 Object Detection with TTS')
    parser.add_argument('--config', type=str, default='config_v8.yaml',
                       help='Configuration file path (default: config_v8.yaml)')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model path from config')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image, video, directory, or camera ID (0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--show', action='store_true',
                       help='Show results in window')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save detection results to CSV')
    parser.add_argument('--tts', action='store_true',
                       help='Enable text-to-speech announcements')
    parser.add_argument('--tts-interval', type=float, default=None,
                       help='Override TTS announcement interval')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logging.warning(f"Configuration file {args.config} not found. Using defaults.")
        config = {
            'model': {'path': 'yolov8n.pt', 'confidence_threshold': 0.25, 'iou_threshold': 0.45, 'max_detections': 1000},
            'tts': {'enabled': False, 'announcement_interval': 5.0, 'speech_rate': 150, 'volume': 0.8},
            'display': {'show_results': False, 'line_thickness': 3, 'hide_labels': False, 'hide_confidence': False},
            'output': {'save_results': True, 'save_csv': False, 'save_txt': False, 'output_dir': 'runs/detect_v8'},
            'performance': {'device': 'auto', 'half_precision': False, 'optimize': True},
            'filtering': {'classes': None, 'agnostic_nms': False, 'augment': False},
            'logging': {'verbose': True, 'log_level': 'INFO', 'save_logs': False}
        }
    
    # Override config with command line arguments
    if args.model:
        config['model']['path'] = args.model
    if args.tts:
        config['tts']['enabled'] = True
    if args.tts_interval:
        config['tts']['announcement_interval'] = args.tts_interval
    if args.show:
        config['display']['show_results'] = True
    if args.save_csv:
        config['output']['save_csv'] = True
    
    # Setup logging
    setup_logging(config)
    
    # Initialize detector
    detector = YOLOv8Detector(config)
    
    # Determine source type and run detection
    source = args.source
    
    if source.isdigit() or source == '0':
        # Webcam
        camera_id = int(source)
        detector.detect_webcam(camera_id, config['display']['show_results'])
    
    elif os.path.isfile(source):
        # Single file (image or video)
        file_ext = Path(source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Image
            logging.info(f"Detecting objects in image: {source}")
            detections = detector.detect_image(source, args.output, config['display']['show_results'])
            logging.info(f"Found {len(detections)} objects")
            for det in detections:
                logging.info(f"  {det['class_name']}: {det['confidence']:.2f}")
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            logging.info(f"Detecting objects in video: {source}")
            detections = detector.detect_video(source, args.output, config['display']['show_results'], config['output']['save_csv'])
            logging.info(f"Total detections: {len(detections)}")
    
    elif os.path.isdir(source):
        # Directory of images
        logging.info(f"Processing directory: {source}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(source).glob(f'*{ext}'))
            image_files.extend(Path(source).glob(f'*{ext.upper()}'))
        
        if not image_files:
            logging.error(f"No image files found in directory: {source}")
            return
        
        logging.info(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            logging.info(f"Processing: {img_path.name}")
            detections = detector.detect_image(str(img_path), None, False)
            logging.info(f"  Found {len(detections)} objects")
    
    else:
        logging.error(f"Invalid source '{source}'. Please provide a valid file, directory, or camera ID.")
        return


if __name__ == "__main__":
    main()
