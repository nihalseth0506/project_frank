"""
generate_training_data.py

Generates synthetic YOLO training data from MuJoCo overhead camera.
Spawns cube at random positions, renders frames, auto-labels bounding boxes.

Output structure:
    data/
        images/train/  ← PNG frames
        labels/train/  ← YOLO .txt labels
        images/val/    ← validation frames
        labels/val/    ← validation labels
        data.yaml      ← YOLO dataset config

Usage:
    python scripts/generate_training_data.py
"""

import os
import sys
import math
import random
import numpy as np
import mujoco
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")
DATA_DIR   = os.path.join(_PROJ_ROOT, "data")

# ── camera parameters ────────────────────────────────────────────────────────
CAM_NAME   = "overhead_cam"
IMG_W      = 640
IMG_H      = 480
FOVY_DEG   = 60.0

# ── camera world position (must match XML) ────────────────────────────────────
CAM_X      = 0.5
CAM_Y      = -0.05
CAM_Z      = 1.5

# ── scene constants ───────────────────────────────────────────────────────────
TABLE_Z    = 0.530
CUBE_HALF  = 0.025
CUBE_Z     = TABLE_Z + CUBE_HALF   # 0.555

# ── cube spawn zone (match pick_place_scripted_env.py) ────────────────────────
OBJECT_LOW  = np.array([0.42, -0.05])
OBJECT_HIGH = np.array([0.62,  0.12])

# retracted pose — arm lifted and pulled back, out of camera view
RETRACTED_QPOS = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853, 0.04, 0.04])
# ── dataset split ─────────────────────────────────────────────────────────────
N_TRAIN    = 800
N_VAL      = 200
HOME_QPOS  = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04])


def world_to_pixel(wx, wy, cam_x, cam_y, cam_z, fovy_deg, img_w, img_h):
    """
    Convert world (x, y) to image pixel (u, v).
    Camera looks straight down (euler 0 0 0 in MuJoCo = looking along -z).
    MuJoCo camera: x=right, y=up in image, z=out of image.
    """
    fovy_rad  = math.radians(fovy_deg)
    half_fov  = fovy_rad / 2.0
    aspect    = img_w / img_h

    height_above_table = cam_z - TABLE_Z

    # scale: how many meters per normalized unit
    scale_y = math.tan(half_fov) * height_above_table
    scale_x = scale_y * aspect

    # offset from camera center in world
    dx = wx - cam_x
    dy = wy - cam_y

    # normalize to [-1, 1]
    # MuJoCo overhead cam: world +x → image right (+u), world +y → image up (-v)
    u_norm =  dx / scale_x
    v_norm = -dy / scale_y

    # convert to pixel
    u = int((u_norm + 1.0) / 2.0 * img_w)
    v = int((v_norm + 1.0) / 2.0 * img_h)

    return u, v


def cube_pixel_size(cam_z, fovy_deg, img_w, img_h):
    """Approximate pixel size of cube face from overhead camera."""
    fovy_rad  = math.radians(fovy_deg)
    half_fov  = fovy_rad / 2.0
    aspect    = img_w / img_h

    height_above = cam_z - TABLE_Z
    scale_y = math.tan(half_fov) * height_above
    scale_x = scale_y * aspect

    # cube is 0.04m wide (2 * CUBE_HALF)
    cube_w_m = CUBE_HALF * 2
    px_w = int(cube_w_m / (2 * scale_x) * img_w)
    px_h = int(cube_w_m / (2 * scale_y) * img_h)

    return px_w, px_h


def generate_dataset():
    model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data   = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    tray_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")

    # precompute cube pixel size
    px_w, px_h = cube_pixel_size(CAM_Z, FOVY_DEG, IMG_W, IMG_H)
    print(f"Cube pixel size: {px_w}w x {px_h}h pixels")

    splits = [("train", N_TRAIN), ("val", N_VAL)]

    for split_name, n_samples in splits:
        img_dir = os.path.join(DATA_DIR, "images", split_name)
        lbl_dir = os.path.join(DATA_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print(f"\nGenerating {n_samples} {split_name} samples...")

        for i in range(n_samples):
            # reset to home
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = RETRACTED_QPOS[:7].copy()
            data.qpos[7]  = 0.04
            data.qpos[8]  = 0.04
            data.ctrl[:7] = RETRACTED_QPOS[:7].copy()
            data.ctrl[7]  = 255.0

            # random cube position in spawn zone
            cx = np.random.uniform(OBJECT_LOW[0], OBJECT_HIGH[0])
            cy = np.random.uniform(OBJECT_LOW[1], OBJECT_HIGH[1])

            # random z rotation for cube
            theta = np.random.uniform(0, 2 * np.pi)
            qw    = np.cos(theta / 2)
            qz    = np.sin(theta / 2)

            qs = 9
            data.qpos[qs:qs+3] = [cx, cy, CUBE_Z]
            data.qpos[qs+3]    = qw
            data.qpos[qs+4]    = 0.0
            data.qpos[qs+5]    = 0.0
            data.qpos[qs+6]    = qz
            data.qvel[qs:qs+6] = 0.0

            # move tray out of spawn zone so it doesn't confuse detection
            tgt_x = np.random.uniform(0.45, 0.55)
            tgt_y = np.random.uniform(-0.25, -0.15)
            model.body_pos[tray_id]   = [tgt_x, tgt_y, TABLE_Z + 0.001]
            model.body_pos[target_id] = [tgt_x, tgt_y, TABLE_Z + 0.002]

            mujoco.mj_forward(model, data)

            # render
            renderer.update_scene(data, camera=CAM_NAME)
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # compute YOLO label from known world position
            u, v = world_to_pixel(cx, cy, CAM_X, CAM_Y, CAM_Z,
                                   FOVY_DEG, IMG_W, IMG_H)

            # YOLO format: class cx cy w h (all normalized 0-1)
            yolo_cx = u / IMG_W
            yolo_cy = v / IMG_H
            yolo_w  = px_w / IMG_W
            yolo_h  = px_h / IMG_H

            # skip if cube is outside image bounds
            if not (0.05 < yolo_cx < 0.95 and 0.05 < yolo_cy < 0.95):
                print(f"  [{i}] cube outside image bounds — skipping")
                continue

            # save image
            img_path = os.path.join(img_dir, f"frame_{i:05d}.png")
            cv2.imwrite(img_path, frame_bgr)

            # save label
            lbl_path = os.path.join(lbl_dir, f"frame_{i:05d}.txt")
            with open(lbl_path, "w") as f:
                f.write(f"0 {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

            if i % 100 == 0:
                print(f"  [{i}/{n_samples}] cube=({cx:.3f},{cy:.3f}) "
                      f"pixel=({u},{v}) yolo=({yolo_cx:.3f},{yolo_cy:.3f})")

    # write data.yaml
    yaml_path = os.path.join(DATA_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATA_DIR}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['cube']\n")

    print(f"\nDataset written to {DATA_DIR}")
    print(f"data.yaml: {yaml_path}")
    print(f"\nNext step: python scripts/train_yolo.py")


if __name__ == "__main__":
    generate_dataset()