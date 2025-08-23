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
from pathlib import Path
from collections import defaultdict
import threading

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


class TTSManager:
    """Manages text-to-speech functionality with time-based announcements."""
    
    def __init__(self, announcement_interval=5.0):
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
                if LOGGER:
                    LOGGER.info(f"Text-to-speech engine initialized successfully (announcement interval: {announcement_interval}s)")
            except Exception as e:
                if LOGGER:
                    LOGGER.warning(f"Failed to initialize text-to-speech engine: {e}")
                self.engine = None
    
    def update_detections(self, detections, current_time):
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
                if LOGGER:
                    LOGGER.info("TTS: No objects detected")
    
    def _announce_detections(self, detections):
        """Announce detected objects in a non-blocking way."""
        # detections is already a defaultdict(int) with counts
        announcement_parts = []
        for obj_name, count in detections.items():
            if count == 1:
                announcement_parts.append(f"1 {obj_name}")
            else:
                announcement_parts.append(f"{count} {obj_name}s")
        
        announcement = f"Detected: {', '.join(announcement_parts)}"
        
        # Log the announcement for debugging
        if LOGGER:
            LOGGER.info(f"TTS Announcement: {announcement}")
        
        # Use threading to avoid blocking the main detection loop
        def speak():
            with self.lock:
                try:
                    self.engine.say(announcement)
                    self.engine.runAndWait()
                    if LOGGER:
                        LOGGER.info(f"TTS announcement completed: {announcement}")
                except Exception as e:
                    if LOGGER:
                        LOGGER.warning(f"TTS announcement failed: {e}")
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()


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
            tts_manager = TTSManager(announcement_interval=tts_interval)
    
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
    
    # Initialize time tracking for TTS
    start_time = time.time()
    
    # Check if source is webcam
    is_webcam = source == "0" or source.isdigit()
    
    if is_webcam:
        # Webcam mode with real-time display and TTS
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            if LOGGER:
                LOGGER.error(f"Failed to open webcam {source}")
            return
        
        # Initialize time tracking for TTS
        start_time = time.time()
        last_tts_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
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
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{names[cls]} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Collect for TTS
                        detected_objects.append(names[cls])
            
            # TTS announcement for webcam
            if tts_manager and detected_objects:
                current_time = time.time() - start_time
                if current_time - last_tts_time >= tts_manager.announcement_interval:
                    tts_manager.update_detections(detected_objects, current_time)
                    last_tts_time = current_time
            
            # Display the frame
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # Regular file/directory mode
        results = model(source, conf=conf_thres, iou=iou_thres, classes=classes, 
                       agnostic_nms=agnostic_nms, max_det=max_det, augment=augment,
                       save=save_img, save_txt=save_txt, save_conf=save_conf,
                       project=project, name=name, exist_ok=exist_ok,
                       line_width=line_thickness, show_labels=not hide_labels,
                       show_conf=not hide_conf, vid_stride=vid_stride)
        
        # Process results for TTS
        for result in results:
            if tts_manager:
                current_time = time.time() - start_time
                
                # Collect detected objects for TTS announcement
                detected_objects = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        detected_objects.append(names[cls])
                
                tts_manager.update_detections(detected_objects, current_time)
        
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    """Execute YOLOv8 model inference."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
