#!/usr/bin/env python3
import argparse
import ctypes
import math
import sys
import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO

# Windows VK for left mouse button
VK_LBUTTON = 0x01

# Windows API functions and constants
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
GetSystemMetrics = ctypes.windll.user32.GetSystemMetrics
SendInput = ctypes.windll.user32.SendInput
GetCursorPos = ctypes.windll.user32.GetCursorPos

# Mouse event constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def is_left_mouse_down() -> bool:
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0

def get_cursor_pos():
    pt = POINT()
    GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def send_relative_mouse_move(dx, dy):
    """Send relative mouse movement to the OS (moves in-game crosshair)."""
    if dx == 0 and dy == 0:
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, None)
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def centered_box(screen_w: int, screen_h: int, fov: int):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

def main():
    parser = argparse.ArgumentParser(description="Instant crosshair lock using relative mouse input.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--fov", type=int, default=350, help="Capture size in pixels")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--aim-height", type=float, default=10.0, help="Aim height factor")
    parser.add_argument("--lock-radius", type=int, default=5, help="Lock radius in pixels")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS")
    args = parser.parse_args()

    try:
        screen_w = GetSystemMetrics(0)
        screen_h = GetSystemMetrics(1)
    except Exception:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            screen_w, screen_h = mon["width"], mon["height"]

    roi = centered_box(screen_w, screen_h, args.fov)
    center_x, center_y = args.fov // 2, args.fov // 2

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}")
        sys.exit(1)

    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] CUDA enabled")
            model.to("cuda")
        else:
            print("[INFO] Running on CPU")
            model.to("cpu")
    except Exception:
        print("[INFO] Torch not found, default device")

    print("[INFO] Ready. Hold LEFT MOUSE BUTTON to aim lock.")
    print("[INFO] Press Ctrl+C to exit.")

    sct = mss.mss()
    last_active = False
    last_print = 0.0

    try:
        while True:
            start_time = time.perf_counter()
            lmb = is_left_mouse_down()

            if not lmb:
                if last_active:
                    print("[STATUS] IDLE (LMB released)")
                    last_active = False
                continue

            if not last_active:
                print("[STATUS] ACTIVE (LMB held)")
                last_active = True

            try:
                raw = sct.grab(roi)
                frame = np.array(raw, dtype=np.uint8)
                if frame.size == 0:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(f"[ERROR] Screen capture failed: {e}")
                continue

            try:
                results = model.predict(
                    source=frame,
                    verbose=False,
                    conf=args.conf,
                    iou=args.iou,
                    half=False,
                    device=None
                )
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                continue

            res = results[0]
            if res.boxes is None or len(res.boxes.xyxy) == 0:
                if time.perf_counter() - last_print > 0.2:
                    print("[TARGET] none")
                    last_print = time.perf_counter()
                continue

            person_boxes = [box for box, cls in zip(res.boxes.xyxy, res.boxes.cls) if int(cls) == 0]
            if not person_boxes:
                if time.perf_counter() - last_print > 0.2:
                    print("[TARGET] no person detected")
                    last_print = time.perf_counter()
                continue

            closest = None
            closest_dist = None
            for box_tensor in person_boxes:
                x1, y1, x2, y2 = map(int, box_tensor.tolist())
                h = y2 - y1
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                head_y = int(my - (h / args.aim_height))
                head_x = mx

                # Heuristic to ignore self/player
                own_player = (x1 < 15) or (x1 < args.fov / 5 and y2 > args.fov / 1.2)
                if own_player:
                    continue

                d = math.hypot(head_x - center_x, head_y - center_y)
                if closest_dist is None or d < closest_dist:
                    closest_dist = d
                    closest = (head_x, head_y)

            if closest is None:
                if time.perf_counter() - last_print > 0.2:
                    print("[TARGET] none (filtered)")
                    last_print = time.perf_counter()
                continue

            hx, hy = closest
            msg = "LOCKED" if closest_dist <= args.lock_radius else "TARGETING"

            # Calculate relative movement from screen center to target
            rel_x = hx - center_x
            rel_y = hy - center_y

            if lmb:
                send_relative_mouse_move(rel_x, rel_y)

            now = time.perf_counter()
            if now - last_print > 0.05:
                if args.show_fps:
                    fps = 1.0 / max(1e-6, (now - start_time))
                    print(f"[{msg}] dist={closest_dist:.1f} fps={fps:.1f} aiming relative move ({rel_x},{rel_y})")
                else:
                    print(f"[{msg}] dist={closest_dist:.1f} aiming relative move ({rel_x},{rel_y})")
                last_print = now


    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    except Exception as e:
        print(f"[FATAL] {e}")
    finally:
        try:
            sct.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()