"""
train_yolo.py
Fine-tunes YOLOv8n on synthetic MuJoCo frames to detect the red cube.

Usage:
    python scripts/train_yolo.py
"""

from ultralytics import YOLO
import os

def main():
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    _PROJ_ROOT = os.path.dirname(_HERE)

    DATA_YAML  = os.path.join(_PROJ_ROOT, "data", "data.yaml")
    MODEL_DIR  = os.path.join(_PROJ_ROOT, "models", "yolo")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")

    model.train(
        data     = DATA_YAML,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        name     = "cube_detector",
        project  = MODEL_DIR,
        device   = "0",
        patience = 10,
        save     = True,
        verbose  = True,
        workers  = 0,        # ← set to 0 on Windows to avoid multiprocessing issues
    )

    print(f"\nBest model: {MODEL_DIR}/cube_detector/weights/best.pt")

if __name__ == "__main__":
    main()