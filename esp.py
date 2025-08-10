#!/usr/bin/env python3
import argparse
import ctypes
import sys
import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO

def centered_box(screen_w: int, screen_h: int, fov: int):
    left = (screen_w - fov) // 2
    top = (screen_h - fov) // 2
    return {"left": left, "top": top, "width": fov, "height": fov}

def draw_esp(frame, box, show_distance, box_color):
    """
    Draw bounding box and optional distance text for one detected person box.
    box: (x1,y1,x2,y2)
    box_color: (B,G,R) tuple
    """
    x1, y1, x2, y2 = box
    h = y2 - y1

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    if show_distance:
        # Approximate distance (arbitrary scaling)
        distance = 1000 / (h + 1)
        dist_text = f"{distance:.1f}m"
        cv2.putText(frame, dist_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def get_user_input(prompt, default, cast_type=str, valid_options=None):
    while True:
        inp = input(f"{prompt} [{default}]: ").strip()
        if inp == "":
            return default
        try:
            val = cast_type(inp)
            if valid_options and val not in valid_options:
                print(f"Invalid option. Choose from: {valid_options}")
                continue
            return val
        except ValueError:
            print(f"Invalid input. Please enter a {cast_type.__name__}.")

def main():
    print("=== ESP Overlay Configuration ===")

    model_path = input("YOLO model path (default yolov8n.pt): ").strip() or "yolov8n.pt"
    fov = get_user_input("FOV (capture size in pixels)", 375, int)
    conf = get_user_input("Confidence threshold (0.0 - 1.0)", 0.4, float)
    iou = get_user_input("IoU threshold (0.0 - 1.0)", 0.45, float)
    show_distance = get_user_input("Show distance on boxes? (yes/no)", "yes", str, ["yes", "no"])
    show_distance = True if show_distance.lower() == "yes" else False

    print("\nChoose box color:")
    print("1 - Green (default)")
    print("2 - Red")
    print("3 - Blue")
    print("4 - Yellow")
    color_choice = get_user_input("Select box color (1-4)", 1, int, [1, 2, 3, 4])
    color_map = {
        1: (0, 255, 0),    # Green
        2: (0, 0, 255),    # Red
        3: (255, 0, 0),    # Blue
        4: (0, 255, 255),  # Yellow
    }
    box_color = color_map[color_choice]

    resizable_input = get_user_input("Make window resizable? (yes/no)", "yes", str, ["yes", "no"])
    window_resizable = True if resizable_input.lower() == "yes" else False

    print("\nStarting ESP overlay with your settings...\n")

    try:
        from ctypes import windll
        GetSystemMetrics = windll.user32.GetSystemMetrics
        screen_w = GetSystemMetrics(0)
        screen_h = GetSystemMetrics(1)
    except Exception:
        with mss.mss() as sct:
            mon = sct.monitors[1]
            screen_w, screen_h = mon["width"], mon["height"]

    roi = centered_box(screen_w, screen_h, fov)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_path}': {e}")
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

    print("[INFO] ESP overlay started. Press ESC to exit.")

    sct = mss.mss()

    window_flag = cv2.WINDOW_NORMAL if window_resizable else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow("ESP Overlay", window_flag)

    try:
        while True:
            start_time = time.perf_counter()
            try:
                raw = sct.grab(roi)
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(f"[ERROR] Screen capture failed: {e}")
                continue

            try:
                results = model.predict(
                    source=frame,
                    verbose=False,
                    conf=conf,
                    iou=iou,
                    half=False,
                    device=None,
                )
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                continue

            res = results[0]
            if res.boxes is None or len(res.boxes.xyxy) == 0:
                cv2.putText(frame, "No targets detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
                    if int(cls) == 0:  # person class
                        box_int = tuple(map(int, box.tolist()))
                        draw_esp(frame, box_int, show_distance, box_color)

            fps = 1.0 / max(time.perf_counter() - start_time, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("ESP Overlay", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")

    finally:
        try:
            sct.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
