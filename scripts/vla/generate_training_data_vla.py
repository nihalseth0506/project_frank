"""
generate_training_data_vla.py

Generates synthetic YOLO training data for Phase 3 VLA.
Spawns all three cubes (red, blue, yellow) in every frame.
Each frame gets three YOLO label lines — one per cube.
YOLO class is always 0 (single class: 'cube') — colour identified by HSV.

Output structure:
    data_vla/
        images/train/
        labels/train/
        images/val/
        labels/val/
        data.yaml

Usage:
    python scripts/vla/generate_training_data_vla.py
"""

import os
import sys
import math
import numpy as np
import mujoco
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))

SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")
DATA_DIR   = os.path.join(_PROJ_ROOT, "data_vla")

# camera parameters — must match XML
CAM_NAME = "overhead_cam"
IMG_W    = 640
IMG_H    = 480
FOVY_DEG = 60.0
CAM_X    = 0.5
CAM_Y    = -0.05
CAM_Z    = 1.5

TABLE_Z   = 0.530
CUBE_HALF = 0.025
CUBE_Z    = TABLE_Z + CUBE_HALF   # 0.555

# spawn zone — wider than Phase 2 to fit 3 cubes
SPAWN_LOW  = np.array([0.42, -0.05])
SPAWN_HIGH = np.array([0.62,  0.12])

MIN_CUBE_DIST = 0.09   # minimum distance between cube centers

# arm retracted pose — out of camera frame
RETRACTED_QPOS = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853, 0.04, 0.04])

# freejoint qpos start indices
QPOS_STARTS = {'red': 9, 'blue': 16, 'yellow': 23}
COLOURS     = ['red', 'blue', 'yellow']

N_TRAIN = 800
N_VAL   = 200


def world_to_pixel(wx, wy):
    fovy_rad     = math.radians(FOVY_DEG)
    half_fov     = fovy_rad / 2.0
    aspect       = IMG_W / IMG_H
    height_above = CAM_Z - TABLE_Z
    scale_y      = math.tan(half_fov) * height_above
    scale_x      = scale_y * aspect

    dx = wx - CAM_X
    dy = wy - CAM_Y

    u_norm =  dx / scale_x
    v_norm = -dy / scale_y

    u = int((u_norm + 1.0) / 2.0 * IMG_W)
    v = int((v_norm + 1.0) / 2.0 * IMG_H)

    return u, v


def cube_pixel_size():
    fovy_rad     = math.radians(FOVY_DEG)
    half_fov     = fovy_rad / 2.0
    aspect       = IMG_W / IMG_H
    height_above = CAM_Z - TABLE_Z
    scale_y      = math.tan(half_fov) * height_above
    scale_x      = scale_y * aspect

    cube_w_m = CUBE_HALF * 2
    px_w     = int(cube_w_m / (2 * scale_x) * IMG_W)
    px_h     = int(cube_w_m / (2 * scale_y) * IMG_H)

    return px_w, px_h


def spawn_no_overlap(n=3, min_dist=MIN_CUBE_DIST):
    """Return n positions with minimum separation."""
    positions = []
    attempts  = 0

    while len(positions) < n and attempts < 500:
        x   = np.random.uniform(SPAWN_LOW[0], SPAWN_HIGH[0])
        y   = np.random.uniform(SPAWN_LOW[1], SPAWN_HIGH[1])
        pos = np.array([x, y])

        if all(np.linalg.norm(pos - p) > min_dist for p in positions):
            positions.append(pos)

        attempts += 1

    if len(positions) < n:
        # fallback fixed positions
        positions = [
            np.array([0.52,  0.08]),
            np.array([0.52, -0.02]),
            np.array([0.44,  0.04]),
        ]

    return positions


def generate_dataset():
    model    = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    tray_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")

    px_w, px_h = cube_pixel_size()
    print(f"Cube pixel size: {px_w}w x {px_h}h pixels")
    print(f"Output: {DATA_DIR}")

    splits = [("train", N_TRAIN), ("val", N_VAL)]

    for split_name, n_samples in splits:
        img_dir = os.path.join(DATA_DIR, "images", split_name)
        lbl_dir = os.path.join(DATA_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print(f"\nGenerating {n_samples} {split_name} samples...")

        frame_idx = 0

        for i in range(n_samples):
            # reset
            mujoco.mj_resetData(model, data)
            data.qpos[:9] = RETRACTED_QPOS.copy()
            data.ctrl[:7] = RETRACTED_QPOS[:7].copy()
            data.ctrl[7]  = 255.0

            # spawn all three cubes with no overlap
            positions = spawn_no_overlap()

            cube_world_pos = {}

            for colour, pos in zip(COLOURS, positions):
                qs    = QPOS_STARTS[colour]
                theta = np.random.uniform(0, 2 * np.pi)
                qw    = np.cos(theta / 2)
                qz    = np.sin(theta / 2)

                data.qpos[qs:qs+3] = [pos[0], pos[1], CUBE_Z]
                data.qpos[qs+3]    = qw
                data.qpos[qs+4]    = 0.0
                data.qpos[qs+5]    = 0.0
                data.qpos[qs+6]    = qz
                data.qvel[qs:qs+6] = 0.0

                cube_world_pos[colour] = pos

            # move tray out of workspace
            model.body_pos[tray_id]   = [0.50, -0.20, TABLE_Z + 0.001]
            model.body_pos[target_id] = [0.50, -0.20, TABLE_Z + 0.002]

            mujoco.mj_forward(model, data)

            # render
            renderer.update_scene(data, camera=CAM_NAME)
            frame     = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # build label — one line per cube
            label_lines = []
            skip_frame  = False

            for colour in COLOURS:
                cx, cy = cube_world_pos[colour]
                u, v   = world_to_pixel(cx, cy)

                yolo_cx = u / IMG_W
                yolo_cy = v / IMG_H
                yolo_w  = px_w / IMG_W
                yolo_h  = px_h / IMG_H

                # skip frame if any cube is outside image bounds
                if not (0.03 < yolo_cx < 0.97 and 0.03 < yolo_cy < 0.97):
                    skip_frame = True
                    break

                label_lines.append(
                    f"0 {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}"
                )

            if skip_frame:
                print(f"  [{i}] cube outside image bounds — skipping")
                continue

            # save image
            img_path = os.path.join(img_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(img_path, frame_bgr)

            # save label
            lbl_path = os.path.join(lbl_dir, f"frame_{frame_idx:05d}.txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")

            if frame_idx % 100 == 0:
                positions_str = " | ".join(
                    f"{c}=({cube_world_pos[c][0]:.2f},{cube_world_pos[c][1]:.2f})"
                    for c in COLOURS
                )
                print(f"  [{frame_idx}/{n_samples}] {positions_str}")

            frame_idx += 1

        print(f"  Saved {frame_idx} {split_name} frames")

    # write data.yaml — single class, colour from HSV not YOLO
    yaml_path = os.path.join(DATA_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATA_DIR}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['cube']\n")

    print(f"\nDataset written to {DATA_DIR}")
    print(f"data.yaml: {yaml_path}")
    print(f"\nNext step: python scripts/vla/train_yolo_vla.py")


if __name__ == "__main__":
    generate_dataset()