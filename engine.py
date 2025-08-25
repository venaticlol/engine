#!/usr/bin/env python3
import ctypes
import math
import time
import threading
import cv2
import mss
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk

# ------------------- Windows API -------------------
VK_RBUTTON = 0x02  # Right mouse button
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
GetSystemMetrics = ctypes.windll.user32.GetSystemMetrics
SendInput = ctypes.windll.user32.SendInput

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),("mi", MOUSEINPUT)]

def is_right_mouse_down():
    return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0

def send_relative_mouse_move(dx, dy):
    if dx == 0 and dy == 0:
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def centered_box(screen_w, screen_h, fov):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

# ------------------- Aimbot Class -------------------
class Aimbot:
    def __init__(self):
        self.screen_w = GetSystemMetrics(0)
        self.screen_h = GetSystemMetrics(1)
        self.model = YOLO("yolov8n.pt")
        self.fov = 350
        self.conf = 0.15
        self.aim_height = 10.0
        self.lock_radius = 5
        self.smooth_factor = 1.0
        self.last_target = None
        self.running = False
        self.show_fps = True

    def run(self):
        self.running = True
        sct = mss.mss()
        last_time = time.perf_counter()

        try:
            while self.running:
                # Check RMB
                rmb_pressed = is_right_mouse_down()
                if not rmb_pressed:
                    time.sleep(0.01)
                    continue

                roi = centered_box(self.screen_w, self.screen_h, self.fov)

                try:
                    raw = sct.grab(roi)
                    frame = np.array(raw, dtype=np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                except:
                    time.sleep(0.01)
                    continue

                try:
                    results = self.model.predict(
                        source=frame,
                        verbose=False,
                        conf=self.conf,
                        iou=0.45,
                        half=False,
                        device=None
                    )
                except:
                    time.sleep(0.01)
                    continue

                res = results[0]
                if res.boxes is None or len(res.boxes.xyxy) == 0:
                    time.sleep(0.01)
                    continue

                person_boxes = [box for box, cls in zip(res.boxes.xyxy, res.boxes.cls) if int(cls) == 0]
                if not person_boxes:
                    time.sleep(0.01)
                    continue

                # Target selection
                closest = None
                closest_dist = float("inf")
                center_x, center_y = self.fov // 2, self.fov // 2
                for box_tensor in person_boxes:
                    x1, y1, x2, y2 = map(int, box_tensor.tolist())
                    h = y2 - y1
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    head_y = int(my - (h / self.aim_height))
                    head_x = mx

                    if self.is_self_player(x1, y1, x2, y2, frame):
                        continue

                    d = math.hypot(head_x - center_x, head_y - center_y)
                    if d < closest_dist:
                        closest_dist = d
                        closest = (head_x, head_y)

                if closest:
                    hx, hy = closest
                    rel_x = hx - center_x
                    rel_y = hy - center_y

                    # Smooth aiming
                    if self.last_target:
                        dx = hx - self.last_target[0]
                        dy = hy - self.last_target[1]
                        if abs(dx) < 1 and abs(dy) < 1:
                            rel_x, rel_y = 0, 0
                        else:
                            rel_x /= self.smooth_factor
                            rel_y /= self.smooth_factor

                    send_relative_mouse_move(rel_x, rel_y)
                    self.last_target = (hx, hy)

                # Update FPS safely
                if self.show_fps:
                    now = time.perf_counter()
                    fps = 1 / max(1e-6, now - last_time)
                    last_time = now
                    try:
                        fps_label.config(text=f"FPS: {fps:.1f}")
                    except tk.TclError:
                        pass

                time.sleep(0.005)  # Tiny sleep to reduce CPU usage

        finally:
            try:
                sct.close()
            except:
                pass
            self.running = False

    def is_self_player(self, x1, y1, x2, y2, frame):
        if x1 < self.fov * 0.2 and y2 > self.fov * 0.8:
            return True

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width > self.fov * 0.6 or box_height > self.fov * 0.6:
            return True

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_bright = np.array([0, 0, 180])
        upper_bright = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_bright, upper_bright)
        bright_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        if bright_ratio > 0.3:
            return True

        return False

    def stop(self):
        self.running = False

# ------------------- GUI -------------------
def start_aimbot():
    if not aimbot.running:
        threading.Thread(target=aimbot.run, daemon=True).start()

def stop_aimbot():
    aimbot.stop()

def update_settings(val):
    aimbot.fov = fov_slider.get()
    aimbot.conf = conf_slider.get() / 100
    aimbot.smooth_factor = max(1, smooth_slider.get() / 10)  # Avoid division by zero

aimbot = Aimbot()

root = tk.Tk()
root.title("sakura.lol")
root.geometry("350x350")
root.configure(bg="#0d0d0d")  # Dark background

style = ttk.Style(root)
style.theme_use("clam")

# Button style
style.configure("TButton", foreground="#ffffff", background="#660000", font=("Arial", 10, "bold"))
style.map("TButton", background=[("active", "#990000"), ("disabled", "#330000")])

style.configure("TLabel", background="#0d0d0d", foreground="#ff4d4d", font=("Arial", 10, "bold"))

# FOV Slider
ttk.Label(root, text="FOV").pack(pady=(10,0))
fov_slider = tk.Scale(root, from_=100, to=1000, orient=tk.HORIZONTAL, length=300,
                      command=update_settings, bg="#330000", fg="#ff4d4d", troughcolor="#660000",
                      highlightbackground="#0d0d0d", activebackground="#990000")
fov_slider.set(350)
fov_slider.pack()

# Confidence Slider
ttk.Label(root, text="Confidence (%)").pack(pady=(10,0))
conf_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, length=300,
                       command=update_settings, bg="#330000", fg="#ff4d4d", troughcolor="#660000",
                       highlightbackground="#0d0d0d", activebackground="#990000")
conf_slider.set(15)
conf_slider.pack()

# Smoothing Slider
ttk.Label(root, text="Smoothing (%)").pack(pady=(10,0))
smooth_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, length=300,
                         command=update_settings, bg="#330000", fg="#ff4d4d", troughcolor="#660000",
                         highlightbackground="#0d0d0d", activebackground="#990000")
smooth_slider.set(10)
smooth_slider.pack()

# Buttons
ttk.Button(root, text="Start", command=start_aimbot).pack(pady=(10,5))
ttk.Button(root, text="Stop", command=stop_aimbot).pack()

# FPS Label
fps_label = ttk.Label(root, text="FPS: 0")
fps_label.pack(pady=(10,0))

root.mainloop()