#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import math
import time
import threading
import cv2
import mss
import numpy as np
from ultralytics import YOLO
import sys
import os
import requests
from pathlib import Path
from filterpy.kalman import KalmanFilter

# ------------------- Windows API -------------------
VK_RBUTTON = 0x02  # Right mouse button
user32 = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState
GetSystemMetrics = user32.GetSystemMetrics
SendInput = user32.SendInput

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

def is_right_mouse_down():
    return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0

def send_relative_mouse_move(dx, dy):
    if dx == 0 and dy == 0:
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def send_absolute_mouse_move(x, y):
    # Convert to absolute coordinates (0-65535)
    screen_w = GetSystemMetrics(0)
    screen_h = GetSystemMetrics(1)
    abs_x = int(x * 65535 / screen_w)
    abs_y = int(y * 65535 / screen_h)
    
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def click():
    # Mouse down
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    
    # Mouse up
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def centered_box(screen_w, screen_h, fov):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

# ------------------- Model Management -------------------
def get_model_path(model_name):
    """Get the path where the model should be stored"""
    home = Path.home()
    model_dir = home / ".sakura_aimbot" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{model_name}.pt"

def is_model_downloaded(model_name):
    """Check if model file exists locally"""
    model_path = get_model_path(model_name)
    return model_path.exists()

def download_model(model_name):
    """Download model from Ultralytics if not present"""
    model_path = get_model_path(model_name)
    
    if is_model_downloaded(model_name):
        return str(model_path)
    
    # Ultralytics model URLs
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    model_urls = {
        "yolov8n": f"{base_url}yolov8n.pt",
        "yolov8s": f"{base_url}yolov8s.pt",
        "yolov8m": f"{base_url}yolov8m.pt",
        "yolov8l": f"{base_url}yolov8l.pt",
        "yolov8x": f"{base_url}yolov8x.pt",
        "yolov5nu": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5nu.pt",
        "yolov5su": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5su.pt",
        "yolov5mu": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5mu.pt",
        "yolov5lu": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5lu.pt",
        "yolov5xu": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5xu.pt"
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Unsupported model: {model_name}")
    
    print(f"Downloading {model_name} model...")
    url = model_urls[model_name]
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise

# ------------------- Aimbot Core -------------------
class Aimbot:
    def __init__(self):
        self.screen_w = GetSystemMetrics(0)
        self.screen_h = GetSystemMetrics(1)
        self.model_name = "yolov8n"  # Default model
        self.model = self.load_model(self.model_name)
        self.fov = 350
        self.conf = 0.15
        self.aim_height = 10.0
        self.lock_radius = 5
        self.smooth_factor = 1.0
        self.last_target = None
        self.running = False
        self.show_fps = True
        self.aim_method = "Smooth Aim"  # Default aim method
        self.target_priority = "Closest to Crosshair"  # Default target priority
        self.sticky_aim = False  # Sticky aim toggle
        self.sticky_target = None  # Current sticky target
        self.sticky_lock_time = 0.3  # Lock time in seconds
        self.sticky_last_lock = 0  # Last time target was locked
        # Pre-allocate arrays for performance
        self.lower_bright = np.array([0, 0, 180], dtype=np.uint8)
        self.upper_bright = np.array([180, 30, 255], dtype=np.uint8)
        # Kalman filter for advanced smoothing
        self.kalman = None
        self.kalman_initialized = False

    def load_model(self, model_name):
        """Load a YOLO model, downloading if necessary"""
        try:
            model_path = download_model(model_name)
            return YOLO(model_path)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to default model
            return YOLO("yolov8n.pt")

    def switch_model(self, model_name):
        """Switch to a different YOLO model"""
        if model_name == self.model_name:
            return True
            
        try:
            self.model = self.load_model(model_name)
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"Failed to switch to model {model_name}: {e}")
            return False

    def is_self_player(self, x1, y1, x2, y2, frame):
        # Heuristics to ignore player's own body/weapon overlay
        if x1 < self.fov * 0.2 and y2 > self.fov * 0.8:
            return True

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width > self.fov * 0.6 or box_height > self.fov * 0.6:
            return True

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # Use pre-allocated arrays
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bright, self.upper_bright)
        bright_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        if bright_ratio > 0.3:
            return True

        return False

# ------------------- PySide6 UI -------------------
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QAction, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QFrame, QSpacerItem, QSizePolicy, QGroupBox,
    QComboBox, QCheckBox, QMessageBox
)

class AimbotWorker(QThread):
    fpsUpdated = Signal(float)
    statusText = Signal(str)
    runningChanged = Signal(bool)

    def __init__(self, core: Aimbot):
        super().__init__()
        self.core = core
        self._stop_flag = threading.Event()
        # Pre-calculate constants
        self.center_x = self.core.fov // 2
        self.center_y = self.core.fov // 2

    def stop(self):
        self._stop_flag.set()
        self.core.running = False

    def run(self):
        self.core.running = True
        self.runningChanged.emit(True)
        self.statusText.emit("Running")
        
        # Use mss with optimized settings
        with mss.mss() as sct:
            # Pre-calculate ROI
            roi = centered_box(self.core.screen_w, self.core.screen_h, self.core.fov)
            
            # Pre-allocate numpy arrays
            last_time = time.perf_counter()
            frame_buffer = None
            
            try:
                while not self._stop_flag.is_set():
                    # Only aim when right mouse is pressed
                    if not is_right_mouse_down():
                        # Reset sticky target when RMB is released
                        self.core.sticky_target = None
                        time.sleep(0.001)  # Reduced sleep time
                        continue

                    try:
                        # Capture screen with minimal processing
                        raw = sct.grab(roi)
                        if frame_buffer is None:
                            frame_buffer = np.empty((raw.height, raw.width, 4), dtype=np.uint8)
                        
                        # Use frombuffer for zero-copy conversion
                        frame = np.frombuffer(raw.rgb, dtype=np.uint8).reshape((raw.height, raw.width, 3))
                    except Exception:
                        time.sleep(0.001)
                        continue

                    try:
                        # Run inference with optimized settings
                        results = self.core.model.predict(
                            source=frame,
                            verbose=False,
                            conf=self.core.conf,
                            iou=0.45,
                            half=True,  # Use half precision for speed
                            device=0 if hasattr(self.core.model, 'names') else None  # GPU if available
                        )
                    except Exception:
                        time.sleep(0.001)
                        continue

                    res = results[0]
                    if res.boxes is None or len(res.boxes.xyxy) == 0:
                        time.sleep(0.001)  # Reduced sleep time
                        # Update FPS
                        now = time.perf_counter()
                        fps = 1 / max(1e-6, now - last_time)
                        last_time = now
                        if self.core.show_fps:
                            self.fpsUpdated.emit(fps)
                        continue

                    # Filter person class (0) with vectorized operations
                    try:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        person_indices = np.where(classes == 0)[0]
                        person_boxes = boxes[person_indices]
                    except Exception:
                        person_boxes = []

                    if len(person_boxes) == 0:
                        time.sleep(0.001)
                        now = time.perf_counter()
                        fps = 1 / max(1e-6, now - last_time)
                        last_time = now
                        if self.core.show_fps:
                            self.fpsUpdated.emit(fps)
                        continue

                    # Target selection (closest to center) - vectorized
                    head_positions = []
                    box_sizes = []
                    for box in person_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        h = y2 - y1
                        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                        head_y = int(my - (h / self.core.aim_height))
                        head_x = mx
                        head_positions.append((head_x, head_y))
                        box_sizes.append((x2 - x1) * (y2 - y1))  # Area of bounding box

                    # Vectorized distance calculation
                    head_positions = np.array(head_positions)
                    distances = np.sqrt(np.sum((head_positions - [self.center_x, self.center_y])**2, axis=1))
                    
                    # Filter out self-player detections
                    valid_indices = []
                    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                        hx, hy = head_positions[i]
                        # Convert back to original coordinates for self-player check
                        if not self.core.is_self_player(int(x1), int(y1), int(x2), int(y2), frame):
                            valid_indices.append(i)
                    
                    if not valid_indices:
                        time.sleep(0.001)
                        now = time.perf_counter()
                        fps = 1 / max(1e-6, now - last_time)
                        last_time = now
                        if self.core.show_fps:
                            self.fpsUpdated.emit(fps)
                        continue

                    # Handle sticky aim
                    current_time = time.time()
                    target_locked = False
                    closest = None
                    
                    if self.core.sticky_aim:
                        # Check if we should keep the current sticky target
                        if self.core.sticky_target is not None:
                            # Check if the sticky target is still valid
                            target_still_valid = False
                            for i in valid_indices:
                                hx, hy = head_positions[i]
                                dist = np.sqrt((hx - self.core.sticky_target[0])**2 + (hy - self.core.sticky_target[1])**2)
                                if dist < self.core.lock_radius:
                                    target_still_valid = True
                                    closest = self.core.sticky_target
                                    target_locked = True
                                    break
                            
                            # If target is no longer valid, release it
                            if not target_still_valid:
                                self.core.sticky_target = None
                                self.core.sticky_last_lock = 0
                        
                        # If no sticky target or it expired, acquire a new one
                        if self.core.sticky_target is None or (current_time - self.core.sticky_last_lock) > self.core.sticky_lock_time:
                            # Apply target priority to select new sticky target
                            if self.core.target_priority == "Closest to Crosshair":
                                # Find closest valid target
                                valid_distances = distances[valid_indices]
                                closest_idx = valid_indices[np.argmin(valid_distances)]
                                closest = head_positions[closest_idx]
                            elif self.core.target_priority == "Largest Target":
                                # Find largest valid target
                                valid_sizes = [box_sizes[i] for i in valid_indices]
                                largest_idx = valid_indices[np.argmax(valid_sizes)]
                                closest = head_positions[largest_idx]
                            elif self.core.target_priority == "Smallest Target":
                                # Find smallest valid target
                                valid_sizes = [box_sizes[i] for i in valid_indices]
                                smallest_idx = valid_indices[np.argmin(valid_sizes)]
                                closest = head_positions[smallest_idx]
                            elif self.core.target_priority == "Closest and Largest":
                                # Combine distance and size (normalized)
                                valid_distances = distances[valid_indices]
                                valid_sizes = [box_sizes[i] for i in valid_indices]
                                # Normalize values
                                norm_distances = valid_distances / np.max(valid_distances)
                                norm_sizes = np.array(valid_sizes) / np.max(valid_sizes)
                                # Combine with weights (70% distance, 30% size)
                                combined_scores = 0.7 * norm_distances + 0.3 * (1 - norm_sizes)
                                best_idx = valid_indices[np.argmin(combined_scores)]
                                closest = head_positions[best_idx]
                            else:  # Default to closest
                                valid_distances = distances[valid_indices]
                                closest_idx = valid_indices[np.argmin(valid_distances)]
                                closest = head_positions[closest_idx]
                            
                            # Set new sticky target
                            self.core.sticky_target = closest
                            self.core.sticky_last_lock = current_time
                            target_locked = True
                    else:
                        # Normal targeting without sticky aim
                        # Apply target priority
                        if self.core.target_priority == "Closest to Crosshair":
                            # Find closest valid target
                            valid_distances = distances[valid_indices]
                            closest_idx = valid_indices[np.argmin(valid_distances)]
                            closest = head_positions[closest_idx]
                        elif self.core.target_priority == "Largest Target":
                            # Find largest valid target
                            valid_sizes = [box_sizes[i] for i in valid_indices]
                            largest_idx = valid_indices[np.argmax(valid_sizes)]
                            closest = head_positions[largest_idx]
                        elif self.core.target_priority == "Smallest Target":
                            # Find smallest valid target
                            valid_sizes = [box_sizes[i] for i in valid_indices]
                            smallest_idx = valid_indices[np.argmin(valid_sizes)]
                            closest = head_positions[smallest_idx]
                        elif self.core.target_priority == "Closest and Largest":
                            # Combine distance and size (normalized)
                            valid_distances = distances[valid_indices]
                            valid_sizes = [box_sizes[i] for i in valid_indices]
                            # Normalize values
                            norm_distances = valid_distances / np.max(valid_distances)
                            norm_sizes = np.array(valid_sizes) / np.max(valid_sizes)
                            # Combine with weights (70% distance, 30% size)
                            combined_scores = 0.7 * norm_distances + 0.3 * (1 - norm_sizes)
                            best_idx = valid_indices[np.argmin(combined_scores)]
                            closest = head_positions[best_idx]
                        else:  # Default to closest
                            valid_distances = distances[valid_indices]
                            closest_idx = valid_indices[np.argmin(valid_distances)]
                            closest = head_positions[closest_idx]

                    if closest is not None:
                        hx, hy = closest
                        rel_x = hx - self.center_x
                        rel_y = hy - self.center_y

                        # Apply different aim methods
                        if self.core.aim_method == "Smooth Aim":
                            if self.core.last_target:
                                dx = hx - self.core.last_target[0]
                                dy = hy - self.core.last_target[1]
                                if abs(dx) < 1 and abs(dy) < 1:
                                    rel_x, rel_y = 0, 0
                                else:
                                    rel_x /= max(1.0, self.core.smooth_factor)
                                    rel_y /= max(1.0, self.core.smooth_factor)
                        elif self.core.aim_method == "Linear Aim":
                            rel_x /= max(1.0, self.core.smooth_factor)
                            rel_y /= max(1.0, self.core.smooth_factor)
                        elif self.core.aim_method == "Exponential Aim":
                            rel_x = (rel_x * (1.0 - math.exp(-abs(rel_x) / self.core.smooth_factor))) if rel_x != 0 else 0
                            rel_y = (rel_y * (1.0 - math.exp(-abs(rel_y) / self.core.smooth_factor))) if rel_y != 0 else 0
                        elif self.core.aim_method == "Cubic Aim":
                            rel_x = (rel_x ** 3) / (self.core.smooth_factor * 1000)
                            rel_y = (rel_y ** 3) / (self.core.smooth_factor * 1000)
                        elif self.core.aim_method == "Quadratic Aim":
                            rel_x = (rel_x ** 2) / (self.core.smooth_factor * 100)
                            rel_y = (rel_y ** 2) / (self.core.smooth_factor * 100)
                        elif self.core.aim_method == "Sinusoidal Aim":
                            rel_x = math.sin(rel_x / self.core.smooth_factor) * 10
                            rel_y = math.sin(rel_y / self.core.smooth_factor) * 10
                        elif self.core.aim_method == "Logarithmic Aim":
                            rel_x = math.log(abs(rel_x) + 1) * (1 if rel_x >= 0 else -1) * self.core.smooth_factor
                            rel_y = math.log(abs(rel_y) + 1) * (1 if rel_y >= 0 else -1) * self.core.smooth_factor
                        elif self.core.aim_method == "Square Root Aim":
                            rel_x = math.sqrt(abs(rel_x)) * (1 if rel_x >= 0 else -1) * self.core.smooth_factor
                            rel_y = math.sqrt(abs(rel_y)) * (1 if rel_y >= 0 else -1) * self.core.smooth_factor
                        elif self.core.aim_method == "Constant Speed":
                            distance = math.sqrt(rel_x**2 + rel_y**2)
                            if distance > 0:
                                rel_x = (rel_x / distance) * self.core.smooth_factor
                                rel_y = (rel_y / distance) * self.core.smooth_factor
                        elif self.core.aim_method == "Jump Aim":
                            # Move directly to target with no smoothing
                            pass
                        elif self.core.aim_method == "Adaptive Aim":
                            distance = math.sqrt(rel_x**2 + rel_y**2)
                            factor = max(1.0, distance / self.core.smooth_factor)
                            rel_x /= factor
                            rel_y /= factor
                        elif self.core.aim_method == "Kalman Filter Aim":
                            # Initialize Kalman filter on first use
                            if not self.core.kalman_initialized:
                                self.core.kalman = KalmanFilter(dim_x=4, dim_z=2)
                                self.core.kalman.x = np.array([hx, hy, 0, 0])  # Initial state
                                self.core.kalman.F = np.array([[1, 0, 1, 0],
                                                              [0, 1, 0, 1],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]])  # State transition
                                self.core.kalman.H = np.array([[1, 0, 0, 0],
                                                              [0, 1, 0, 0]])  # Measurement function
                                self.core.kalman.P *= 1000  # Covariance
                                self.core.kalman.R *= 5     # Measurement noise
                                self.core.kalman.Q *= 0.1   # Process noise
                                self.core.kalman_initialized = True
                            
                            # Predict and update
                            self.core.kalman.predict()
                            self.core.kalman.update([hx, hy])
                            predicted_x, predicted_y = self.core.kalman.x[:2]
                            
                            # Calculate movement with smoothing
                            rel_x = (predicted_x - self.center_x) / max(1.0, self.core.smooth_factor)
                            rel_y = (predicted_y - self.center_y) / max(1.0, self.core.smooth_factor)
                        elif self.core.aim_method == "PID Controller Aim":
                            # Simple PID controller implementation
                            if not hasattr(self.core, 'pid_x'):
                                self.core.pid_x = {'prev_error': 0, 'integral': 0}
                                self.core.pid_y = {'prev_error': 0, 'integral': 0}
                            
                            # PID constants
                            Kp, Ki, Kd = 0.5, 0.1, 0.05
                            
                            # X-axis PID
                            error_x = rel_x
                            self.core.pid_x['integral'] += error_x
                            derivative_x = error_x - self.core.pid_x['prev_error']
                            output_x = Kp * error_x + Ki * self.core.pid_x['integral'] + Kd * derivative_x
                            self.core.pid_x['prev_error'] = error_x
                            
                            # Y-axis PID
                            error_y = rel_y
                            self.core.pid_y['integral'] += error_y
                            derivative_y = error_y - self.core.pid_y['prev_error']
                            output_y = Kp * error_y + Ki * self.core.pid_y['integral'] + Kd * derivative_y
                            self.core.pid_y['prev_error'] = error_y
                            
                            rel_x = output_x / max(1.0, self.core.smooth_factor)
                            rel_y = output_y / max(1.0, self.core.smooth_factor)
                        elif self.core.aim_method == "Bezier Curve Aim":
                            # Bezier curve interpolation for smooth movement
                            if self.core.last_target:
                                # Control points for bezier curve
                                start = np.array(self.core.last_target)
                                end = np.array([hx, hy])
                                control = (start + end) / 2  # Simple midpoint control
                                
                                # Calculate point on bezier curve (t=0.5 for mid-point smoothing)
                                t = 0.5
                                bezier_point = (1-t)**2 * start + 2*(1-t)*t * control + t**2 * end
                                rel_x = (bezier_point[0] - self.center_x) / max(1.0, self.core.smooth_factor)
                                rel_y = (bezier_point[1] - self.center_y) / max(1.0, self.core.smooth_factor)
                            else:
                                rel_x /= max(1.0, self.core.smooth_factor)
                                rel_y /= max(1.0, self.core.smooth_factor)

                        send_relative_mouse_move(rel_x, rel_y)
                        self.core.last_target = (hx, hy)

                    # FPS update
                    now = time.perf_counter()
                    fps = 1 / max(1e-6, now - last_time)
                    last_time = now
                    if self.core.show_fps:
                        self.fpsUpdated.emit(fps)

                    # Minimal sleep to prevent 100% CPU usage
                    time.sleep(0.0001)

            finally:
                self.core.running = False
                self.runningChanged.emit(False)
                self.statusText.emit("Stopped")

# ------------------- Modern UI -------------------
class ValueBadge(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setObjectName("ValueBadge")
        self.setAlignment(Qt.AlignCenter)

class StatusPill(QLabel):
    def __init__(self, text="Idle"):
        super().__init__(text)
        self.setObjectName("StatusPill")
        self.setAlignment(Qt.AlignCenter)

    def set_status(self, text, running: bool):
        self.setText(text)
        self.setProperty("running", running)
        self.style().unpolish(self)
        self.style().polish(self)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Core
        self.aimbot = Aimbot()
        self.worker: AimbotWorker | None = None

        # Window
        self.setWindowTitle("sakura aimbot UI")
        self.setMinimumSize(440, 685)

        # Icon (gracefully handle if not present)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakura.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Central widget + layout
        cw = QWidget()
        root = QVBoxLayout(cw)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        header = self._build_header()
        sliders = self._build_sliders()
        controls = self._build_controls()

        root.addWidget(header)
        root.addWidget(sliders)
        root.addWidget(controls)

        # Footer
        footer = QHBoxLayout()
        self.fpsLabel = QLabel("FPS: 0.0")
        self.fpsLabel.setObjectName("Subtle")
        footer.addWidget(self.fpsLabel)

        footer.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        credit = QLabel("Developed by: Japan")
        credit.setObjectName("Subtle")
        footer.addWidget(credit)
        root.addLayout(footer)

        self.setCentralWidget(cw)

        # Shortcuts
        QShortcut(QKeySequence("F8"), self, activated=self.toggle_start_stop)
        QShortcut(QKeySequence("Esc"), self, activated=self.stop_clicked)

        # Theme
        self.setStyleSheet(self._qss())

    def _build_header(self) -> QWidget:
        box = QFrame()
        box.setObjectName("HeaderCard")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        title = QLabel("Sakura Aimbot UI")
        title.setObjectName("H1")
        subtitle = QLabel("• RMB to activate")
        subtitle.setObjectName("Subtle")

        status_row = QHBoxLayout()
        status_row.setSpacing(8)

        self.statusPill = StatusPill("Idle")
        self.statusPill.set_status("Idle", False)

        status_row.addWidget(self.statusPill)
        status_row.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addLayout(status_row)
        return box

    def _build_sliders(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(16)

        # FOV
        self.fovLabel = QLabel("FOV")
        self.fovSlider = self._slider(100, 1000, self.aimbot.fov, self.on_fov_changed)
        self.fovBadge = ValueBadge(str(self.aimbot.fov))

        # Model Selection
        self.modelLabel = QLabel("Model")
        self.modelCombo = QComboBox()
        models = [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu"
        ]
        self.modelCombo.addItems(models)
        self.modelCombo.setCurrentText(self.aimbot.model_name)
        self.modelCombo.currentTextChanged.connect(self.on_model_changed)

        # Aim Method Dropdown
        self.aimMethodLabel = QLabel("Aim Method")
        self.aimMethodCombo = QComboBox()
        aim_methods = [
            "Smooth Aim", "Linear Aim", "Exponential Aim", "Cubic Aim",
            "Quadratic Aim", "Sinusoidal Aim", "Logarithmic Aim",
            "Square Root Aim", "Constant Speed", "Jump Aim", "Adaptive Aim",
            "Kalman Filter Aim", "PID Controller Aim", "Bezier Curve Aim"
        ]
        self.aimMethodCombo.addItems(aim_methods)
        self.aimMethodCombo.setCurrentText(self.aimbot.aim_method)
        self.aimMethodCombo.currentTextChanged.connect(self.on_aim_method_changed)

        # Target Priority Dropdown
        self.targetPriorityLabel = QLabel("Target Priority")
        self.targetPriorityCombo = QComboBox()
        target_priorities = [
            "Closest to Crosshair", "Largest Target", "Smallest Target", 
            "Closest and Largest"
        ]
        self.targetPriorityCombo.addItems(target_priorities)
        self.targetPriorityCombo.setCurrentText(self.aimbot.target_priority)
        self.targetPriorityCombo.currentTextChanged.connect(self.on_target_priority_changed)

        # Confidence (%)
        self.confLabel = QLabel("Confidence (%)")
        conf_percent = int(self.aimbot.conf * 100)
        self.confSlider = self._slider(1, 100, conf_percent, self.on_conf_changed)
        self.confBadge = ValueBadge(str(conf_percent))

        # Smoothing (%)
        self.smoothLabel = QLabel("Smoothing (%)")
        smooth_percent = int(self.aimbot.smooth_factor * 10)  # inverse mapping from original
        if smooth_percent < 1: smooth_percent = 10
        self.smoothSlider = self._slider(1, 100, smooth_percent, self.on_smooth_changed)
        self.smoothBadge = ValueBadge(str(smooth_percent))

        # Aim Height (head offset divisor)
        self.aimHeightLabel = QLabel("Aim Height Divisor")
        aim_div = int(self.aimbot.aim_height)
        self.aimHeightSlider = self._slider(3, 30, aim_div, self.on_aim_height_changed)
        self.aimHeightBadge = ValueBadge(str(aim_div))

        # Sticky Aim Toggle
        self.stickyAimLabel = QLabel("Sticky Aim")
        self.stickyAimCheckbox = QCheckBox()
        self.stickyAimCheckbox.setChecked(self.aimbot.sticky_aim)
        self.stickyAimCheckbox.stateChanged.connect(self.on_sticky_aim_changed)

        # Layout grid
        row = 0
        grid.addWidget(self.fovLabel, row, 0)
        grid.addWidget(self.fovSlider, row, 1)
        grid.addWidget(self.fovBadge, row, 2)
        row += 1

        grid.addWidget(self.modelLabel, row, 0)
        grid.addWidget(self.modelCombo, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.aimMethodLabel, row, 0)
        grid.addWidget(self.aimMethodCombo, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.targetPriorityLabel, row, 0)
        grid.addWidget(self.targetPriorityCombo, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.confLabel, row, 0)
        grid.addWidget(self.confSlider, row, 1)
        grid.addWidget(self.confBadge, row, 2)
        row += 1

        grid.addWidget(self.smoothLabel, row, 0)
        grid.addWidget(self.smoothSlider, row, 1)
        grid.addWidget(self.smoothBadge, row, 2)
        row += 1

        grid.addWidget(self.aimHeightLabel, row, 0)
        grid.addWidget(self.aimHeightSlider, row, 1)
        grid.addWidget(self.aimHeightBadge, row, 2)
        row += 1

        grid.addWidget(self.stickyAimLabel, row, 0)
        grid.addWidget(self.stickyAimCheckbox, row, 1)
        row += 1

        return card

    def _build_controls(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.startBtn = QPushButton("Start")
        self.startBtn.setObjectName("Accent")
        self.startBtn.clicked.connect(self.start_clicked)

        self.stopBtn = QPushButton("Stop")
        self.stopBtn.clicked.connect(self.stop_clicked)
        self.stopBtn.setEnabled(False)

        btn_row.addWidget(self.startBtn)
        btn_row.addWidget(self.stopBtn)
        btn_row.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        helper = QLabel("• Press F8 to toggle start/stop • Press Esc to stop\n• RMB must be held to aim")
        helper.setObjectName("Hint")

        lay.addLayout(btn_row)
        lay.addWidget(helper)
        return card

    def _slider(self, mn, mx, val, on_change):
        s = QSlider(Qt.Horizontal)
        s.setRange(mn, mx)
        s.setValue(val)
        s.setSingleStep(1)
        s.valueChanged.connect(on_change)
        s.setObjectName("Slider")
        return s

    # ------------------- Slots: slider updates -------------------
    @Slot(int)
    def on_fov_changed(self, v: int):
        self.aimbot.fov = int(v)
        self.fovBadge.setText(str(v))

    @Slot(str)
    def on_model_changed(self, model_name: str):
        # Show a warning dialog before switching models
        reply = QMessageBox.question(
            self, "Model Change", 
            f"Switch to {model_name}? This may take a moment if the model needs to be downloaded.",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            success = self.aimbot.switch_model(model_name)
            if not success:
                QMessageBox.critical(
                    self, 
                    "Model Error", 
                    f"Failed to load model {model_name}. Check your internet connection and try again."
                )
                # Revert to previous model
                self.modelCombo.setCurrentText(self.aimbot.model_name)
        else:
            # Revert to previous model
            self.modelCombo.setCurrentText(self.aimbot.model_name)

    @Slot(int)
    def on_conf_changed(self, v: int):
        self.aimbot.conf = max(0.01, v / 100.0)
        self.confBadge.setText(str(v))

    @Slot(int)
    def on_smooth_changed(self, v: int):
        # Match your original: UI 1..100 maps to factor = max(1, v/10)
        self.aimbot.smooth_factor = max(1.0, v / 10.0)
        self.smoothBadge.setText(str(v))

    @Slot(int)
    def on_aim_height_changed(self, v: int):
        self.aimbot.aim_height = float(v)
        self.aimHeightBadge.setText(str(v))
        
    @Slot(str)
    def on_aim_method_changed(self, method: str):
        self.aimbot.aim_method = method

    @Slot(str)
    def on_target_priority_changed(self, priority: str):
        self.aimbot.target_priority = priority

    @Slot(int)
    def on_sticky_aim_changed(self, state: int):
        self.aimbot.sticky_aim = (state == Qt.Checked)

    # ------------------- Controls -------------------
    @Slot()
    def start_clicked(self):
        if self.worker and self.worker.isRunning():
            return
        self.aimbot.last_target = None
        self.worker = AimbotWorker(self.aimbot)
        self.worker.fpsUpdated.connect(self.update_fps)
        self.worker.statusText.connect(self.update_status)
        self.worker.runningChanged.connect(self.on_running_changed)
        self.worker.start()

    @Slot()
    def stop_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        else:
            self.update_status("Stopped")

    @Slot()
    def toggle_start_stop(self):
        if self.worker and self.worker.isRunning():
            self.stop_clicked()
        else:
            self.start_clicked()

    @Slot(float)
    def update_fps(self, fps: float):
        self.fpsLabel.setText(f"FPS: {fps:.1f}")

    @Slot(str)
    def update_status(self, text: str):
        running = (text.lower() == "running")
        self.statusPill.set_status(text, running)

    @Slot(bool)
    def on_running_changed(self, running: bool):
        self.startBtn.setEnabled(not running)
        self.stopBtn.setEnabled(True if running else False)
        if not running:
            # ensure stop button disables after thread exit
            self.stopBtn.setEnabled(False)

    # ------------------- Styling -------------------
    def _qss(self) -> str:
        # Tailored neon red/black modern style
        return """
        * { 
            color: #fce6e6; 
            font-family: 'Segoe UI', 'Inter', 'Ubuntu', sans-serif;
            font-size: 14px;
        }
        QMainWindow {
            background: #0d0d0f;
        }
        #H1 {
            font-size: 20px;
            font-weight: 700;
            color: #ff4d4d;
        }
        #Subtle {
            color: #bdaaaa;
        }
        #Hint {
            color: #c7b7b7;
            font-size: 12px;
        }
        QFrame#HeaderCard, QFrame#Card {
            background: #161618;
            border: 1px solid #2a2a2e;
            border-radius: 16px;
        }
        QPushButton {
            background: #2a2a2e;
            border: 1px solid #34343a;
            padding: 10px 14px;
            border-radius: 12px;
        }
        QPushButton:hover {
            background: #323238;
        }
        QPushButton:pressed {
            background: #2a2a2e;
            border-color: #3a3a42;
        }
        QPushButton#Accent {
            background: #7f0f0f;
            border-color: #9a1b1b;
            color: #ffffff;
        }
        QPushButton#Accent:hover {
            background: #951616;
        }

        QLabel#ValueBadge, QLabel#StatusPill {
            padding: 6px 10px;
            border-radius: 999px;
            background: #241416;
            border: 1px solid #3a1d21;
            color: #ffb3b3;
            font-weight: 600;
        }

        QLabel#StatusPill[running="true"] {
            background: #112015;
            border: 1px solid #1f3b26;
            color: #b6f3c1;
        }
        QLabel#StatusPill[running="false"] {
            background: #241416;
            border: 1px solid #3a1d21;
            color: #ffb3b3;
        }

        QSlider::groove:horizontal {
            border: 1px solid #2e2e33;
            height: 8px;
            background: #1b1b1f;
            border-radius: 999px;
        }
        QSlider::handle:horizontal {
            background: #ff4d4d;
            border: 1px solid #ff7070;
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
        QSlider::sub-page:horizontal {
            background: #6d1313;
            border: 1px solid #7d1717;
            height: 8px;
            border-radius: 999px;
        }
        QSlider::add-page:horizontal {
            background: #131317;
            border: 1px solid transparent;
            height: 8px;
            border-radius: 999px;
        }
        
        QComboBox {
            background: #2a2a2e;
            border: 1px solid #34343a;
            padding: 8px 12px;
            border-radius: 12px;
            color: #fce6e6;
        }
        QComboBox:hover {
            background: #323238;
        }
        QComboBox::drop-down {
            border: none;
            border-radius: 12px;
        }
        QComboBox QAbstractItemView {
            background: #161618;
            border: 1px solid #2a2a2e;
            selection-background-color: #7f0f0f;
            border-radius: 8px;
        }
        
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #34343a;
            background: #2a2a2e;
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background: #7f0f0f;
            border: 1px solid #9a1b1b;
        }
        
        QMessageBox {
            background: #0d0d0f;
        }
        """

# ------------------- Entry -------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("sakura.lol")

    # Try to set app icon globally if present
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakura.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    w = MainWindow()
    w.resize(520, 745)
    w.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()