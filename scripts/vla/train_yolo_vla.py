"""
train_yolo_vla.py

Fine-tunes YOLOv8n on the VLA synthetic dataset (all three cubes per frame).
Single class: 'cube' — colour classification handled by HSV in pick_place_scripted_env_vla.py.

Output: models/yolo/cube_detector_vla/weights/best.pt

Usage:
    python scripts/vla/train_yolo_vla.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))

DATA_YAML  = os.path.join(_PROJ_ROOT, "data_vla", "data.yaml")
MODEL_DIR  = os.path.join(_PROJ_ROOT, "models", "yolo", "cube_detector_vla")


def main():
    from ultralytics import YOLO

    print("=" * 55)
    print("FRANK Phase 3 — YOLO VLA Training")
    print("Dataset : data_vla/")
    print("Output  : models/yolo/cube_detector_vla/")
    print("=" * 55)

    model = YOLO("yolov8n.pt")

    model.train(
        data       = DATA_YAML,
        epochs     = 50,
        imgsz      = 640,
        batch      = 16,
        lr0        = 0.001,
        patience   = 10,
        project    = os.path.join(_PROJ_ROOT, "models", "yolo"),
        name       = "cube_detector_vla",
        exist_ok   = True,
        workers    = 0,         # required on Windows
        device     = 0,         # GPU
        verbose    = True,
    )

    best_pt = os.path.join(MODEL_DIR, "weights", "best.pt")
    print(f"\nBest model: {best_pt}")


if __name__ == "__main__":
    main()