#!/usr/bin/env python3
import argparse
import ctypes
import sys
import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO

# Windows API constants and functions
SendInput = ctypes.windll.user32.SendInput
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
VK_RBUTTON = 0x02  # Right mouse button
INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

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

def send_mouse_down():
    inp = INPUT(type=INPUT_MOUSE,
                mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None))
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def send_mouse_up():
    inp = INPUT(type=INPUT_MOUSE,
                mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None))
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def centered_box(screen_w: int, screen_h: int, fov: int):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

def main():
    parser = argparse.ArgumentParser(description="Triggerbot hold left click while detecting target and RMB held")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--fov", type=int, default=265, help="Capture size in pixels")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--debounce-time", type=float, default=0.1, help="Seconds to confirm detection")
    parser.add_argument("--min-box-width", type=int, default=30, help="Minimum width of box to count")
    parser.add_argument("--debug", action="store_true", help="Show debug window")
    args = parser.parse_args()

    try:
        from ctypes import windll
        GetSystemMetrics = windll.user32.GetSystemMetrics
        screen_w = GetSystemMetrics(0)
        screen_h = GetSystemMetrics(1)
    except Exception:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            screen_w, screen_h = mon["width"], mon["height"]

    roi = centered_box(screen_w, screen_h, args.fov)

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}")
        sys.exit(1)

    try:
        import torch
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                print("[INFO] Running on GPU")
            except Exception as e:
                print(f"[WARN] GPU move failed: {e}")
                model.to("cpu")
                print("[INFO] Running on CPU")
        else:
            print("[INFO] Running on CPU")
            model.to("cpu")
    except ImportError:
        print("[INFO] PyTorch not installed or no GPU, running on default")

    print("[INFO] Triggerbot started. Hold RIGHT MOUSE BUTTON to activate. Ctrl+C to exit.")

    sct = mss.mss()
    detection_start_time = None
    holding_click = False

    try:
        while True:
            start_time = time.perf_counter()

            # Check if right mouse button (RMB) is held
            rmb_held = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0

            if not rmb_held:
                # RMB not held - release mouse if held, skip detection
                if holding_click:
                    send_mouse_up()
                    holding_click = False
                    print("[STATUS] RMB released, stopping triggerbot and releasing mouse.")
                time.sleep(0.01)
                continue

            # RMB held - proceed with screen capture and detection

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
            person_boxes = []
            if res.boxes is not None and len(res.boxes.xyxy) > 0:
                for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    if int(cls) == 0 and conf >= args.conf:
                        width = box[2] - box[0]
                        if width >= args.min_box_width:
                            person_boxes.append(box)

            now = time.perf_counter()

            if len(person_boxes) == 0:
                # No target detected
                detection_start_time = None
                if holding_click:
                    send_mouse_up()
                    holding_click = False
                    print("[STATUS] Target lost, released mouse button.")

                if args.debug:
                    cv2.putText(frame, "Target: None", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Triggerbot Debug", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                time.sleep(0.01)
                continue

            # Target(s) detected
            if detection_start_time is None:
                detection_start_time = now

            stable_detection = (now - detection_start_time) >= args.debounce_time

            if stable_detection:
                if not holding_click:
                    send_mouse_down()
                    holding_click = True
                    print("[STATUS] Target detected - holding mouse down.")
            else:
                if holding_click:
                    send_mouse_up()
                    holding_click = False
                    print("[STATUS] Debouncing - released mouse button.")

            if args.debug:
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {1/(max(time.perf_counter()-start_time,1e-6)):.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Triggerbot Debug", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        if holding_click:
            send_mouse_up()
        try:
            sct.close()
        except Exception:
            pass
        if args.debug:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()