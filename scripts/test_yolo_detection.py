"""
test_yolo_detection.py
Tests YOLO detection + pixel-to-world transform on a single episode frame.

Usage:
    python scripts/test_yolo_detection.py
"""

import os, sys, math
import numpy as np
import mujoco
import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

SCENE_PATH  = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")
YOLO_PATH   = os.path.join(_PROJ_ROOT, "models", "yolo", "cube_detector", "weights", "best.pt")

TABLE_Z  = 0.530
CUBE_Z   = TABLE_Z + 0.025
CAM_NAME = "overhead_cam"
IMG_W, IMG_H = 640, 480
FOVY_DEG = 60.0
CAM_X, CAM_Y, CAM_Z = 0.5, -0.05, 1.5

HOME_QPOS      = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04])
RETRACTED_QPOS = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853])

OBJECT_LOW  = np.array([0.50, -0.02])
OBJECT_HIGH = np.array([0.60,  0.10])


def pixel_to_world(u, v):
    fovy_rad = math.radians(FOVY_DEG)
    half_fov = fovy_rad / 2.0
    aspect   = IMG_W / IMG_H

    height_above = CAM_Z - TABLE_Z
    scale_y = math.tan(half_fov) * height_above
    scale_x = scale_y * aspect

    u_norm =  (u / IMG_W) * 2.0 - 1.0
    v_norm =  (v / IMG_H) * 2.0 - 1.0

    wx = CAM_X + u_norm * scale_x
    wy = CAM_Y - v_norm * scale_y

    return wx, wy


def main():
    model    = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    detector = YOLO(YOLO_PATH)

    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # spawn cube at known position
    cx = np.random.uniform(OBJECT_LOW[0], OBJECT_HIGH[0])
    cy = np.random.uniform(OBJECT_LOW[1], OBJECT_HIGH[1])

    mujoco.mj_resetData(model, data)
    data.qpos[:7] = RETRACTED_QPOS
    data.qpos[7]  = 0.04
    data.qpos[8]  = 0.04
    data.ctrl[:7] = RETRACTED_QPOS
    data.ctrl[7]  = 255.0

    qs = 9
    data.qpos[qs:qs+3] = [cx, cy, CUBE_Z]
    data.qpos[qs+3]    = 1.0
    data.qpos[qs+4:qs+7] = 0.0
    mujoco.mj_forward(model, data)

    # render overhead frame
    renderer.update_scene(data, camera=CAM_NAME)
    frame     = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # add this before running YOLO, after renderer.render()
    cv2.imwrite("debug_frame.png", frame_bgr)
    print("Saved debug_frame.png — check if cube is visible")
    # run YOLO
    results = detector(frame, verbose=False, conf=0.1)[0]
    print(f"Detections at conf=0.01: {len(results.boxes)}")
    for box in results.boxes:
        print(f"  conf={float(box.conf[0]):.3f} xyxy={box.xyxy[0].tolist()}")

    if len(results.boxes) == 0:
        print("No detection!")
        return

    # take highest confidence detection
    box  = results.boxes[results.boxes.conf.argmax()]
    u    = float(box.xywh[0][0])
    v    = float(box.xywh[0][1])
    conf = float(box.conf[0])

    # convert to world
    wx, wy = pixel_to_world(u, v)

    # ground truth
    gt_x = data.xpos[object_id][0]
    gt_y = data.xpos[object_id][1]

    print(f"Ground truth : x={gt_x:.4f}  y={gt_y:.4f}")
    print(f"YOLO detected: x={wx:.4f}  y={wy:.4f}  conf={conf:.3f}")
    print(f"Error        : dx={abs(wx-gt_x)*100:.1f}mm  dy={abs(wy-gt_y)*100:.1f}mm")

    # draw on frame
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.circle(frame_bgr, (int(u), int(v)), 4, (0,0,255), -1)
    cv2.putText(frame_bgr, f"conf={conf:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite("yolo_detection_test.png", frame_bgr)
    print("Saved yolo_detection_test.png")


if __name__ == "__main__":
    main()