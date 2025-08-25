# esp.py (modified)
import sys
import json
import argparse
import time
import cv2
import mss
import numpy as np
from ultralytics import YOLO

def main():
    # ... [existing config input code] ...

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        sys.exit(1)

    sct = mss.mss()
    roi = {"left": (screen_w - fov) // 2, "top": (screen_h - fov) // 2, "width": fov, "height": fov}

    while True:
        try:
            raw = sct.grab(roi)
            frame = np.array(raw)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(json.dumps({"error": f"Screen capture failed: {e}"}))
            continue

        try:
            results = model.predict(source=frame, verbose=False, conf=conf, iou=iou)
        except Exception as e:
            print(json.dumps({"error": f"Inference failed: {e}"}))
            continue

        boxes = []
        if results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.tolist())
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Output boxes as JSON
        print(json.dumps({"boxes": boxes}))
        sys.stdout.flush()
        time.sleep(0.03)  # ~30 FPS

if __name__ == "__main__":
    main()
