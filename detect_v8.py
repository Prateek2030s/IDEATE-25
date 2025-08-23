# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect_v8.py --weights yolov8n.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import csv
import os
import platform
import sys
import time
import math
from pathlib import Path
from collections import defaultdict
import threading
import queue

import torch
import cv2
import numpy as np

# Try to import pyttsx3 for text-to-speech functionality
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    LOGGER = None  # Will be defined later

# Try to import OCR functionality
OCR_AVAILABLE = False
OCR_ENGINE = None

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_ENGINE = "easyocr"
except ImportError:
    try:
        import pytesseract
        from PIL import Image
        OCR_AVAILABLE = True
        OCR_ENGINE = "pytesseract"
    except ImportError:
        OCR_AVAILABLE = False
        OCR_ENGINE = None

# Import YOLOv8
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    LOGGER = None

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class SpatialDetector:
    """Handles spatial positioning and distance estimation of detected objects."""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Define zones for position detection
        self.zones = {
            'front': {'x_range': (self.center_x - 100, self.center_x + 100), 'y_range': (0, self.center_y)},
            'front_left': {'x_range': (0, self.center_x - 100), 'y_range': (0, self.center_y)},
            'front_right': {'x_range': (self.center_x + 100, frame_width), 'y_range': (0, self.center_y)},
            'left': {'x_range': (0, self.center_x - 100), 'y_range': (self.center_y, frame_height)},
            'right': {'x_range': (self.center_x + 100, frame_width), 'y_range': (self.center_y, frame_height)},
            'back': {'x_range': (self.center_x - 100, self.center_x + 100), 'y_range': (self.center_y, frame_height)}
        }
        
        # Distance estimation parameters (calibrated for typical webcam)
        self.focal_length = 500  # pixels (typical webcam focal length)
        self.known_object_width = 0.5  # meters (average human width)
        self.known_object_height = 1.7  # meters (average human height)
    
    def get_position_description(self, x1, y1, x2, y2):
        """Get relative position description for an object."""
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine which zone the object is in
        for zone_name, zone in self.zones.items():
            if (zone['x_range'][0] <= center_x <= zone['x_range'][1] and 
                zone['y_range'][0] <= center_y <= zone['y_range'][1]):
                
                # Convert zone names to user-friendly descriptions
                position_map = {
                    'front': 'right in front of you',
                    'front_left': 'to your front left',
                    'front_right': 'to your front right',
                    'left': 'to your left',
                    'right': 'to your right',
                    'back': 'behind you'
                }
                return position_map.get(zone_name, zone_name)
        
        return 'in your field of view'
    
    def estimate_distance(self, x1, y1, x2, y2, object_name):
        """Estimate distance to object based on bounding box size."""
        # Calculate bounding box dimensions
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        # Use width or height depending on object type
        if object_name in ['person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
            # Use height for living beings
            distance = (self.known_object_height * self.focal_length) / height_pixels
        else:
            # Use width for objects
            distance = (self.known_object_width * self.focal_length) / width_pixels
        
        # Clamp distance to reasonable range (0.5 to 20 meters)
        distance = max(0.5, min(20.0, distance))
        
        return distance
    
    def format_distance(self, distance):
        """Format distance for speech output."""
        if distance < 1.0:
            return f"{int(distance * 100)} centimeters"
        elif distance < 10.0:
            return f"{distance:.1f} meters"
        else:
            return f"{int(distance)} meters"


class WindowsTTSManager:
    """Windows-specific TTS manager with better threading support and spatial awareness."""
    
    def __init__(self, announcement_interval=5.0):
        self.engine = None
        self.lock = threading.Lock()
        self.announcement_interval = announcement_interval
        self.current_detections = {}
        self.start_time = time.time()
        self.last_announcement_time = 0
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.spatial_detector = None
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                # Windows-specific settings
                if platform.system() == "Windows":
                    # Try to use SAPI5 engine for better Windows compatibility
                    voices = self.engine.getProperty('voices')
                    if voices:
                        self.engine.setProperty('voice', voices[0].id)
                
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.8)
                
                # Start speech thread
                self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
                self.speech_thread.start()
                
                if LOGGER:
                    LOGGER.info(f"Windows TTS engine initialized successfully (announcement interval: {announcement_interval}s)")
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Failed to initialize TTS engine: {e}")
                self.engine = None
    
    def set_spatial_detector(self, spatial_detector):
        """Set the spatial detector for position and distance estimation."""
        self.spatial_detector = spatial_detector
    
    def _speech_worker(self):
        """Background thread for handling speech synthesis."""
        while True:
            try:
                announcement = self.speech_queue.get(timeout=1)
                if announcement is None:  # Shutdown signal
                    break
                
                with self.lock:
                    self.is_speaking = True
                    try:
                        self.engine.say(announcement)
                        self.engine.runAndWait()
                        if LOGGER:
                            LOGGER.info(f"TTS announcement completed: {announcement}")
                    except Exception as e:
                        if LOGGER:
                            LOGGER.warning(f"TTS announcement failed: {e}")
                    finally:
                        self.is_speaking = False
                        
            except queue.Empty:
                continue
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Speech worker error: {e}")
    
    def update_detections(self, detections_with_positions, current_time=None):
        """Update current detections and announce if interval has passed."""
        if not self.engine:
            return
        
        # Use absolute time if not provided
        if current_time is None:
            current_time = time.time() - self.start_time
            
        # Convert detections list to count dictionary with positions
        new_detections = {}
        for detection_info in detections_with_positions:
            if isinstance(detection_info, str):
                # Simple string detection (fallback)
                new_detections[detection_info] = new_detections.get(detection_info, 0) + 1
            else:
                # Structured detection with position info
                obj_name = detection_info['name']
                position = detection_info['position']
                distance = detection_info['distance']
                key = f"{obj_name}_{position}_{distance}"
                new_detections[key] = new_detections.get(key, 0) + 1
        
        # Check if it's time to announce and if detections have changed
        time_since_last = current_time - self.last_announcement_time
        detections_changed = new_detections != self.current_detections
        
        # Announce if: interval has passed AND detections changed, OR this is the first detection
        should_announce = (
            (time_since_last >= self.announcement_interval and detections_changed) or
            (not self.current_detections and new_detections)  # First detection
        )
        
        if should_announce:
            self.current_detections = new_detections.copy()
            self.last_announcement_time = current_time
            
            if new_detections:
                self._announce_detections(detections_with_positions)
            else:
                if LOGGER:
                    LOGGER.info("TTS: No objects detected")
    
    def _announce_detections(self, detections_with_positions):
        """Queue announcement for speech synthesis with spatial information."""
        if not detections_with_positions:
            return
        
        # Group detections by position
        position_groups = defaultdict(list)
        for detection_info in detections_with_positions:
            if isinstance(detection_info, str):
                # Simple string detection (fallback)
                position_groups['in view'].append(detection_info)
            else:
                # Structured detection with position info
                obj_name = detection_info['name']
                position = detection_info['position']
                distance = detection_info['distance']
                position_groups[position].append((obj_name, distance))
        
        # Create announcement parts
        announcement_parts = []
        for position, objects in position_groups.items():
            if position == 'in view':
                # Handle simple detections
                obj_counts = defaultdict(int)
                for obj in objects:
                    obj_counts[obj] += 1
                
                for obj_name, count in obj_counts.items():
                    if count == 1:
                        announcement_parts.append(f"1 {obj_name}")
                    else:
                        announcement_parts.append(f"{count} {obj_name}s")
            else:
                # Handle spatial detections
                obj_counts = defaultdict(list)
                for obj_name, distance in objects:
                    obj_counts[obj_name].append(distance)
                
                for obj_name, distances in obj_counts.items():
                    count = len(distances)
                    avg_distance = sum(distances) / len(distances)
                    distance_str = self.spatial_detector.format_distance(avg_distance) if self.spatial_detector else ""
                    
                    if count == 1:
                        announcement_parts.append(f"1 {obj_name} {position} {distance_str}")
                    else:
                        announcement_parts.append(f"{count} {obj_name}s {position} {distance_str}")
        
        announcement = f"Detected: {', '.join(announcement_parts)}"
        
        if LOGGER:
            LOGGER.info(f"TTS Announcement: {announcement}")
        
        # Queue the announcement for the speech worker
        try:
            self.speech_queue.put(announcement, timeout=0.1)
        except queue.Full:
            if LOGGER:
                LOGGER.warning("TTS queue full, skipping announcement")


class OCRManager:
    """Manages OCR functionality for reading text from images."""
    
    def __init__(self):
        self.reader = None
        self.ocr_engine = OCR_ENGINE
        
        if OCR_AVAILABLE and OCR_ENGINE:
            try:
                if OCR_ENGINE == "pytesseract":
                    # Configure pytesseract path for Windows
                    if platform.system() == "Windows":
                        # You may need to install tesseract and set the path
                        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                        pass
                    self.reader = "pytesseract"
                elif OCR_ENGINE == "easyocr":
                    # Use EasyOCR
                    self.reader = easyocr.Reader(['en'])
                
                if LOGGER:
                    LOGGER.info(f"OCR engine initialized: {self.ocr_engine}")
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Failed to initialize OCR engine: {e}")
                self.reader = None
    
    def extract_text(self, image):
        """Extract text from image using OCR."""
        if not self.reader:
            return []
        
        try:
            if self.ocr_engine == "pytesseract":
                # Convert to PIL Image for pytesseract
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = image
                
                text = pytesseract.image_to_string(pil_image)
                # Split into lines and filter out empty lines
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return lines
            else:
                # Use EasyOCR
                results = self.reader.readtext(image)
                texts = []
                for (bbox, text, prob) in results:
                    if prob > 0.5:  # Confidence threshold
                        texts.append(text.strip())
                return texts
                
        except Exception as e:
            if LOGGER:
                LOGGER.warning(f"OCR extraction failed: {e}")
            return []


def run(
    weights=ROOT / "yolov8n.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    tts=False,  # enable text-to-speech announcements
    tts_interval=5.0,  # TTS announcement interval in seconds
    ocr=False,  # enable OCR text detection
    ocr_interval=3.0,  # OCR check interval in seconds
    camera=0,  # camera index to use
    no_display=False,  # disable video display
    spatial_awareness=True,  # enable spatial awareness and distance estimation
):
    """
    Runs YOLOv8 detection inference on various sources like images, videos, directories, streams, etc.
    """
    # Initialize TTS manager if requested
    tts_manager = None
    if tts:
        if not TTS_AVAILABLE:
            if LOGGER:
                LOGGER.warning("Text-to-speech requested but pyttsx3 is not available. Install with: pip install pyttsx3")
        else:
            tts_manager = WindowsTTSManager(announcement_interval=tts_interval)
    
    # Initialize OCR manager if requested
    ocr_manager = None
    if ocr:
        if not OCR_AVAILABLE:
            if LOGGER:
                LOGGER.warning("OCR requested but OCR libraries not available. Install with: pip install easyocr or pip install pytesseract pillow")
        else:
            ocr_manager = OCRManager()
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    
    # Directories
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    if save_txt:
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(weights)
    if device:
        model.to(device)
    
    # Get class names
    names = model.names
    
    # Check if source is webcam
    is_webcam = source == "0" or source.isdigit()
    
    if is_webcam:
        # Webcam mode with real-time display, TTS, and OCR
        webcam_index = camera  # Use the camera parameter instead of source
        
        # Try to find available cameras
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret:
                    available_cameras.append(i)
                    if LOGGER:
                        LOGGER.info(f"Found camera {i}: {test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                test_cap.release()
        
        if LOGGER:
            LOGGER.info(f"Available cameras: {available_cameras}")
        
        # Use the specified camera or default to first available
        if webcam_index in available_cameras:
            camera_to_use = webcam_index
        elif available_cameras:
            camera_to_use = available_cameras[0]
            if LOGGER:
                LOGGER.warning(f"Camera {webcam_index} not found, using camera {camera_to_use} instead")
        else:
            if LOGGER:
                LOGGER.error("No cameras found!")
            return
        
        cap = cv2.VideoCapture(camera_to_use)
        if not cap.isOpened():
            if LOGGER:
                LOGGER.error(f"Failed to open webcam {camera_to_use}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if LOGGER:
            LOGGER.info(f"Successfully opened camera {camera_to_use}")
            LOGGER.info(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            LOGGER.info(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Initialize spatial detector
        spatial_detector = None
        if spatial_awareness:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            spatial_detector = SpatialDetector(frame_width, frame_height)
            if tts_manager:
                tts_manager.set_spatial_detector(spatial_detector)
            if LOGGER:
                LOGGER.info(f"Spatial detector initialized for {frame_width}x{frame_height} resolution")
        
        last_ocr_time = 0
        frame_count = 0
        display_available = not no_display
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if LOGGER:
                    LOGGER.warning("Failed to read frame from camera")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames (about once per second)
                if LOGGER:
                    LOGGER.info(f"Processing frame {frame_count}, frame shape: {frame.shape}")
                
            # Run inference on current frame
            results = model(frame, conf=conf_thres, iou=iou_thres, classes=classes, 
                          agnostic_nms=agnostic_nms, max_det=max_det, augment=augment,
                          verbose=False)
            
            # Process results and draw bounding boxes
            annotated_frame = frame.copy()
            detected_objects = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates and class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        obj_name = names[cls]
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label with position and distance if spatial awareness is enabled
                        if spatial_awareness and spatial_detector:
                            position = spatial_detector.get_position_description(x1, y1, x2, y2)
                            distance = spatial_detector.estimate_distance(x1, y1, x2, y2, obj_name)
                            distance_str = spatial_detector.format_distance(distance)
                            label = f"{obj_name} {position} {distance_str} {conf:.2f}"
                            
                            # Add position and distance info to detection
                            detected_objects.append({
                                'name': obj_name,
                                'position': position,
                                'distance': distance,
                                'confidence': conf
                            })
                        else:
                            label = f"{obj_name} {conf:.2f}"
                            detected_objects.append(obj_name)
                        
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # TTS announcement for webcam
            if tts_manager:
                tts_manager.update_detections(detected_objects)
            
            # OCR text detection for webcam
            if ocr_manager:
                current_time = time.time()
                if current_time - last_ocr_time >= ocr_interval:
                    detected_texts = ocr_manager.extract_text(frame)
                    if detected_texts:
                        if LOGGER:
                            LOGGER.info(f"OCR detected text: {detected_texts}")
                        # Add OCR text to TTS announcement
                        if tts_manager:
                            ocr_announcement = f"Text detected: {', '.join(detected_texts[:3])}"  # Limit to first 3 texts
                            try:
                                tts_manager.speech_queue.put(ocr_announcement, timeout=0.1)
                            except queue.Full:
                                pass
                    last_ocr_time = current_time
            
            # Display the frame (if display is enabled and available)
            if display_available:
                try:
                    cv2.imshow('YOLOv8 Detection', annotated_frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    if LOGGER:
                        LOGGER.warning(f"Display error: {e}")
                        LOGGER.info("Continuing without display. Press Ctrl+C to stop.")
                    display_available = False
            else:
                # If no display, just add a small delay to prevent 100% CPU usage
                time.sleep(0.01)
                
                # Check for keyboard interrupt to stop
                try:
                    pass
                except KeyboardInterrupt:
                    if LOGGER:
                        LOGGER.info("Stopping detection...")
                    break
        
        cap.release()
        if display_available:
            cv2.destroyAllWindows()
        
    else:
        # Regular file/directory mode
        results = model(source, conf=conf_thres, iou=iou_thres, classes=classes, 
                       agnostic_nms=agnostic_nms, max_det=max_det, augment=augment,
                       save=save_img, save_txt=save_txt, save_conf=save_conf,
                       project=project, name=name, exist_ok=exist_ok,
                       line_width=line_thickness, show_labels=not hide_labels,
                       show_conf=not hide_conf, vid_stride=vid_stride)
        
        # Process results for TTS and OCR
        for result in results:
            if tts_manager or ocr_manager:
                # Collect detected objects for TTS announcement
                detected_objects = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        detected_objects.append(names[cls])
                
                if tts_manager:
                    tts_manager.update_detections(detected_objects)
                
                # OCR for images/videos
                if ocr_manager and hasattr(result, 'orig_img'):
                    detected_texts = ocr_manager.extract_text(result.orig_img)
                    if detected_texts:
                        if LOGGER:
                            LOGGER.info(f"OCR detected text: {detected_texts}")
                        # Add OCR text to TTS announcement
                        if tts_manager:
                            ocr_announcement = f"Text detected: {', '.join(detected_texts[:3])}"
                            try:
                                tts_manager.speech_queue.put(ocr_announcement, timeout=0.1)
                            except queue.Full:
                                pass
        
        # Print results
        if LOGGER:
            LOGGER.info(f"Results saved to {save_dir}")
    
    if update:
        pass  # update model functionality removed for simplicity


def parse_opt():
    """Parse command-line arguments for YOLOv8 detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov8n.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="save format (0=YOLO, 1=Pascal-VOC)")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--tts", action="store_true", help="enable text-to-speech announcements for detected objects")
    parser.add_argument("--tts-interval", type=float, default=5.0, help="TTS announcement interval in seconds (default: 5.0)")
    parser.add_argument("--ocr", action="store_true", help="enable OCR text detection from images/videos")
    parser.add_argument("--ocr-interval", type=float, default=3.0, help="OCR check interval in seconds (default: 3.0)")
    parser.add_argument("--camera", type=int, default=0, help="camera index to use (default: 0)")
    parser.add_argument("--no-display", action="store_true", help="disable video display window (useful when OpenCV GUI is not available)")
    parser.add_argument("--no-spatial", action="store_true", help="disable spatial awareness and distance estimation")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    """Execute YOLOv8 model inference."""
    # Set spatial awareness based on command line argument
    spatial_awareness = not opt.no_spatial
    
    # Remove no_spatial from opt to avoid passing it to run function
    opt_dict = vars(opt)
    opt_dict.pop('no_spatial', None)
    
    run(spatial_awareness=spatial_awareness, **opt_dict)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
