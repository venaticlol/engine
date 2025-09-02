#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
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

# ------------------- Windows API -------------------
VK_RBUTTON = 0x02  # Right mouse button
VK_LBUTTON = 0x01  # Left mouse button
user32 = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState
GetSystemMetrics = user32.GetSystemMetrics
SendInput = user32.SendInput

INPUT_MOUSE = 0
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

def is_left_mouse_down():
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0

def click():
    # Mouse down
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    
    # Small delay
    time.sleep(0.01)
    
    # Mouse up
    inp.mi.dwFlags = MOUSEEVENTF_LEFTUP
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def centered_box(screen_w, screen_h, fov):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

# ------------------- Model Management -------------------
def get_model_path(model_name):
    """Get the path where the model should be stored"""
    home = Path.home()
    model_dir = home / ".triggerbot" / "models"
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

# ------------------- Triggerbot Core -------------------
class Triggerbot:
    def __init__(self):
        self.screen_w = GetSystemMetrics(0)
        self.screen_h = GetSystemMetrics(1)
        self.model_name = "yolov8n"  # Default model
        self.model = self.load_model(self.model_name)
        self.fov = 350
        self.conf = 0.15
        self.trigger_delay = 50  # ms
        self.trigger_key = "right_mouse"  # or "left_mouse" or "always"
        self.lock_radius = 5
        self.target_priority = "closest"  # New: target selection priority
        self.last_target = None
        self.running = False
        self.show_fps = True
        # Pre-allocate arrays for performance
        self.lower_bright = np.array([0, 0, 180], dtype=np.uint8)
        self.upper_bright = np.array([180, 30, 255], dtype=np.uint8)
        
        # Knife check features
        self.knife_check_enabled = False
        self.knife_template = None
        self.knife_template_path = None
        self.load_knife_template()

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

    def set_target_priority(self, priority):
        """Set target selection priority"""
        self.target_priority = priority

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

    def load_knife_template(self):
        """Load the knife equipped template image"""
        try:
            # Look for equipped.png in the same directory as the script
            script_dir = Path(__file__).parent
            template_path = script_dir / "equipped.png"
            
            if template_path.exists():
                self.knife_template_path = str(template_path)
                # Load in grayscale for faster template matching
                self.knife_template = cv2.imread(self.knife_template_path, cv2.IMREAD_GRAYSCALE)
                print(f"Knife template loaded from {template_path}")
            else:
                print(f"Warning: Knife template not found at {template_path}")
        except Exception as e:
            print(f"Error loading knife template: {e}")

    def is_knife_equipped(self, frame):
        """Check if knife is equipped by looking for equipped.png on screen"""
        if not self.knife_check_enabled or self.knife_template is None:
            return False
            
        try:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray_frame, self.knife_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # If match confidence is above threshold, knife is equipped
            return max_val > 0.8  # 80% confidence threshold
        except Exception as e:
            print(f"Error in knife detection: {e}")
            return False

    def set_knife_check(self, enabled):
        """Enable or disable knife check"""
        self.knife_check_enabled = enabled
        if enabled and self.knife_template is None:
            self.load_knife_template()

# ------------------- PySide6 UI -------------------
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QAction, QFont, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QFrame, QSpacerItem, QSizePolicy, QGroupBox,
    QComboBox, QCheckBox, QMessageBox, QButtonGroup, QRadioButton
)

class TriggerbotWorker(QThread):
    fpsUpdated = Signal(float)
    statusText = Signal(str)
    runningChanged = Signal(bool)

    def __init__(self, core: Triggerbot):
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
            
            # Knife detection variables
            knife_detected = False
            
            try:
                while not self._stop_flag.is_set():
                    # Check trigger key condition
                    trigger_condition = (
                        (self.core.trigger_key == "right_mouse" and is_right_mouse_down()) or
                        (self.core.trigger_key == "left_mouse" and is_left_mouse_down()) or
                        (self.core.trigger_key == "always")
                    )
                    
                    # If knife check is enabled, check for knife first
                    if self.core.knife_check_enabled:
                        try:
                            # Capture full screen for knife detection
                            full_screen = {"top": 0, "left": 0, "width": self.core.screen_w, "height": self.core.screen_h}
                            knife_screen = sct.grab(full_screen)
                            knife_frame = np.frombuffer(knife_screen.rgb, dtype=np.uint8).reshape((knife_screen.height, knife_screen.width, 3))
                            
                            # Check if knife is equipped
                            knife_detected = self.core.is_knife_equipped(knife_frame)
                            
                            # If knife detected, skip aimbot functionality
                            if knife_detected:
                                self.statusText.emit("Knife Equipped - Paused")
                                time.sleep(0.1)  # Check again in 100ms
                                continue
                        except Exception as e:
                            pass  # Continue if knife detection fails
                    
                    if not trigger_condition:
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

                    # Target selection based on priority
                    head_positions = []
                    box_sizes = []
                    confidences = []

                    for box in person_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        h = y2 - y1
                        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                        head_y = int(my - (h / 10.0))
                        head_x = mx
                        head_positions.append((head_x, head_y))
                        box_sizes.append((x2-x1) * (y2-y1))
                    
                    # Get confidences if available
                    if hasattr(res.boxes, 'conf') and res.boxes.conf is not None:
                        confidences = res.boxes.conf.cpu().numpy()[person_indices]
                    else:
                        confidences = [1.0] * len(person_boxes)

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

                    # Select target based on priority
                    if self.core.target_priority == "closest":
                        # Find closest valid target
                        valid_distances = distances[valid_indices]
                        closest_idx = valid_indices[np.argmin(valid_distances)]
                        closest = head_positions[closest_idx]
                    elif self.core.target_priority == "largest":
                        # Find largest valid target
                        valid_sizes = [box_sizes[i] for i in valid_indices]
                        largest_idx = valid_indices[np.argmax(valid_sizes)]
                        closest = head_positions[largest_idx]
                    elif self.core.target_priority == "highest_conf":
                        # Find highest confidence valid target
                        valid_confs = [confidences[i] for i in valid_indices]
                        highest_conf_idx = valid_indices[np.argmax(valid_confs)]
                        closest = head_positions[highest_conf_idx]
                    else:  # Default to closest
                        valid_distances = distances[valid_indices]
                        closest_idx = valid_indices[np.argmin(valid_distances)]
                        closest = head_positions[closest_idx]

                    if closest is not None:
                        hx, hy = closest
                        
                        # Check if target is within lock radius
                        distance_to_center = np.sqrt((hx - self.center_x)**2 + (hy - self.center_y)**2)
                        
                        if distance_to_center <= self.core.lock_radius:
                            # Add trigger delay
                            time.sleep(self.core.trigger_delay / 1000.0)
                            click()

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
        self.triggerbot = Triggerbot()
        self.worker: TriggerbotWorker | None = None

        # Window
        self.setWindowTitle("triggerbot Configuration UI")
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
        credit = QLabel("Developed by: godtier")
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

        title = QLabel("Triggerbot Configuration UI")
        title.setObjectName("H1")
        subtitle = QLabel("• Da-Hood based trigger-bot")
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
        self.fovSlider = self._slider(100, 1000, self.triggerbot.fov, self.on_fov_changed)
        self.fovBadge = ValueBadge(str(self.triggerbot.fov))

        # Model Selection
        self.modelLabel = QLabel("Model")
        self.modelCombo = QComboBox()
        models = [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu"
        ]
        self.modelCombo.addItems(models)
        self.modelCombo.setCurrentText(self.triggerbot.model_name)
        self.modelCombo.currentTextChanged.connect(self.on_model_changed)

        # Confidence (%)
        self.confLabel = QLabel("Confidence (%)")
        conf_percent = int(self.triggerbot.conf * 100)
        self.confSlider = self._slider(1, 100, conf_percent, self.on_conf_changed)
        self.confBadge = ValueBadge(str(conf_percent))

        # Trigger Delay (ms)
        self.delayLabel = QLabel("Trigger Delay (ms)")
        self.delaySlider = self._slider(0, 500, self.triggerbot.trigger_delay, self.on_delay_changed)
        self.delayBadge = ValueBadge(str(self.triggerbot.trigger_delay))

        # Lock Radius
        self.radiusLabel = QLabel("Trigger Distance")
        self.radiusSlider = self._slider(1, 50, self.triggerbot.lock_radius, self.on_radius_changed)
        self.radiusBadge = ValueBadge(str(self.triggerbot.lock_radius))

        # Target Priority
        self.priorityLabel = QLabel("Target Priority")
        self.priorityCombo = QComboBox()
        priorities = ["closest", "largest", "highest_conf"]
        self.priorityCombo.addItems(priorities)
        self.priorityCombo.setCurrentText(self.triggerbot.target_priority)
        self.priorityCombo.currentTextChanged.connect(self.on_priority_changed)

        # Trigger Key
        self.triggerKeyLabel = QLabel("Trigger Key")
        self.triggerKeyGroup = QButtonGroup()
        self.triggerKeyLayout = QHBoxLayout()
        
        self.rightMouseRadio = QRadioButton("Right Mouse")
        self.leftMouseRadio = QRadioButton("Left Mouse")
        self.alwaysRadio = QRadioButton("Always Active")
        
        self.triggerKeyGroup.addButton(self.rightMouseRadio)
        self.triggerKeyGroup.addButton(self.leftMouseRadio)
        self.triggerKeyGroup.addButton(self.alwaysRadio)
        
        if self.triggerbot.trigger_key == "right_mouse":
            self.rightMouseRadio.setChecked(True)
        elif self.triggerbot.trigger_key == "left_mouse":
            self.leftMouseRadio.setChecked(True)
        else:
            self.alwaysRadio.setChecked(True)
            
        self.rightMouseRadio.toggled.connect(self.on_trigger_key_changed)
        self.leftMouseRadio.toggled.connect(self.on_trigger_key_changed)
        self.alwaysRadio.toggled.connect(self.on_trigger_key_changed)
        
        self.triggerKeyLayout.addWidget(self.rightMouseRadio)
        self.triggerKeyLayout.addWidget(self.leftMouseRadio)
        self.triggerKeyLayout.addWidget(self.alwaysRadio)

        # Knife Check
        self.knifeCheckLabel = QLabel("Knife Check")
        self.knifeCheckCheckbox = QCheckBox("Enable")
        self.knifeCheckCheckbox.setChecked(self.triggerbot.knife_check_enabled)
        self.knifeCheckCheckbox.stateChanged.connect(self.on_knife_check_changed)

        # Layout grid
        row = 0
        grid.addWidget(self.fovLabel, row, 0)
        grid.addWidget(self.fovSlider, row, 1)
        grid.addWidget(self.fovBadge, row, 2)
        row += 1

        grid.addWidget(self.modelLabel, row, 0)
        grid.addWidget(self.modelCombo, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.confLabel, row, 0)
        grid.addWidget(self.confSlider, row, 1)
        grid.addWidget(self.confBadge, row, 2)
        row += 1

        grid.addWidget(self.delayLabel, row, 0)
        grid.addWidget(self.delaySlider, row, 1)
        grid.addWidget(self.delayBadge, row, 2)
        row += 1

        grid.addWidget(self.radiusLabel, row, 0)
        grid.addWidget(self.radiusSlider, row, 1)
        grid.addWidget(self.radiusBadge, row, 2)
        row += 1

        grid.addWidget(self.priorityLabel, row, 0)
        grid.addWidget(self.priorityCombo, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.triggerKeyLabel, row, 0)
        grid.addLayout(self.triggerKeyLayout, row, 1, 1, 2)
        row += 1

        grid.addWidget(self.knifeCheckLabel, row, 0)
        grid.addWidget(self.knifeCheckCheckbox, row, 1, 1, 2)

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

        helper = QLabel("• Press F8 to toggle start/stop • Press Esc to stop")
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
        self.triggerbot.fov = int(v)
        self.fovBadge.setText(str(v))

    @Slot(str)
    def on_model_changed(self, model_name: str):
        # Show a warning dialog before switching models
        reply = QMessageBox.question(
            self, 
            "Model Change", 
            f"Switch to {model_name}? This may take a moment if the model needs to be downloaded.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.triggerbot.switch_model(model_name)
            if not success:
                QMessageBox.critical(
                    self, 
                    "Model Error", 
                    f"Failed to load model {model_name}. Check your internet connection and try again."
                )
                # Revert to previous model
                self.modelCombo.setCurrentText(self.triggerbot.model_name)
        else:
            # Revert to previous model
            self.modelCombo.setCurrentText(self.triggerbot.model_name)

    @Slot(int)
    def on_conf_changed(self, v: int):
        self.triggerbot.conf = max(0.01, v / 100.0)
        self.confBadge.setText(str(v))

    @Slot(int)
    def on_delay_changed(self, v: int):
        self.triggerbot.trigger_delay = v
        self.delayBadge.setText(str(v))

    @Slot(int)
    def on_radius_changed(self, v: int):
        self.triggerbot.lock_radius = v
        self.radiusBadge.setText(str(v))
        
    @Slot(str)
    def on_priority_changed(self, priority: str):
        self.triggerbot.set_target_priority(priority)
        
    @Slot()
    def on_trigger_key_changed(self):
        if self.rightMouseRadio.isChecked():
            self.triggerbot.trigger_key = "right_mouse"
        elif self.leftMouseRadio.isChecked():
            self.triggerbot.trigger_key = "left_mouse"
        else:
            self.triggerbot.trigger_key = "always"

    @Slot(int)
    def on_knife_check_changed(self, state: int):
        enabled = state == Qt.Checked
        self.triggerbot.set_knife_check(enabled)
        if enabled and self.triggerbot.knife_template is None:
            QMessageBox.warning(
                self,
                "Knife Template Missing",
                "equipped.png not found. Please place the template image in the same directory as this script."
            )

    # ------------------- Controls -------------------
    @Slot()
    def start_clicked(self):
        if self.worker and self.worker.isRunning():
            return
        self.triggerbot.last_target = None
        self.worker = TriggerbotWorker(self.triggerbot)
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
        qss = """
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
        QComboBox::down-arrow {
            image: url(none);
            width: 0px;
            height: 0px;
        }
        QComboBox QAbstractItemView {
            background: #161618;
            border: 1px solid #2a2a2e;
            selection-background-color: #7f0f0f;
            border-radius: 8px;
        }
        
        QRadioButton {
            spacing: 5px;
        }
        
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }
        
        QRadioButton::indicator:unchecked {
            border: 2px solid #555;
            border-radius: 8px;
            background-color: #2a2a2e;
        }
        
        QRadioButton::indicator:checked {
            border: 2px solid #ff4d4d;
            border-radius: 8px;
            background-color: #ff4d4d;
        }
        
        QCheckBox {
            spacing: 5px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        
        QCheckBox::indicator:unchecked {
            border: 2px solid #555;
            border-radius: 4px;
            background-color: #2a2a2e;
        }
        
        QCheckBox::indicator:checked {
            border: 2px solid #ff4d4d;
            border-radius: 4px;
            background-color: #ff4d4d;
        }
        
        QMessageBox {
            background: #0d0d0f;
        }
        """
        return qss

# ------------------- Entry -------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("triggerbot configuration UI")

    w = MainWindow()
    w.resize(520, 785)
    w.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
