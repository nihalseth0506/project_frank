"""
vision.py

Vision module for FRANK Phase 3 VLA.
Handles YOLO detection, HSV colour classification,
pixel-to-world transform, and camera window updates.
"""

import cv2
import math
import numpy as np


def classify_colour_hsv(crop_rgb):
    """
    Classify cube colour from RGB crop using HSV hue.
    Returns 'red', 'blue', 'yellow', or 'unknown'.
    """
    if crop_rgb.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    sat = np.median(hsv[:, :, 1])

    if sat < 50:
        return 'unknown'

    hue = np.median(hsv[:, :, 0])

    if hue < 10 or hue > 160:
        return 'red'
    if 20 < hue < 38:
        return 'yellow'
    if 95 < hue < 135:
        return 'blue'

    return 'unknown'


def pixel_to_world(u, v, img_w, img_h, fovy_deg, cam_x, cam_y, cam_z, table_z):
    """
    Convert pixel (u, v) to world (x, y) using perspective projection.
    Camera looks straight down, z-axis aligned with world -z.
    """
    fovy_rad     = math.radians(fovy_deg)
    half_fov     = fovy_rad / 2.0
    aspect       = img_w / img_h
    height_above = cam_z - table_z
    scale_y      = math.tan(half_fov) * height_above
    scale_x      = scale_y * aspect

    u_norm =  (u / img_w) * 2.0 - 1.0
    v_norm =  (v / img_h) * 2.0 - 1.0

    wx = cam_x + u_norm * scale_x
    wy = cam_y - v_norm * scale_y

    return wx, wy


def detect_cube_pos(renderer, detector, data, cam_name,
                    img_w, img_h, fovy_deg, cam_x, cam_y, cam_z,
                    table_z, cube_z, target_colour, target_object_id,
                    window_name="FRANK — Overhead Camera"):
    """
    Render overhead frame, run YOLO, classify each box by HSV colour,
    return world position of the target colour cube.
    Falls back to ground truth if target colour not detected.
    """
    renderer.update_scene(data, camera=cam_name)
    frame   = renderer.render()
    results = detector(frame, verbose=False, conf=0.1)[0]

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame_bgr, f"Target: {target_colour}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    best_box  = None
    best_conf = 0.0

    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        crop   = frame[y1:y2, x1:x2]
        colour = classify_colour_hsv(crop)
        conf   = float(box.conf[0])
        print(f"    YOLO box: classified={colour} conf={conf:.3f}")

        is_target  = (colour == target_colour)
        colour_bgr = (0, 255, 0) if is_target else (120, 120, 120)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)
        cv2.putText(frame_bgr, f"{colour} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1)

        if is_target and conf > best_conf:
            best_box  = box
            best_conf = conf

    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)

    if best_box is None:
        print(f"  [{target_colour}] not detected — GT fallback")
        return data.xpos[target_object_id].copy()

    u, v = float(best_box.xywh[0][0]), float(best_box.xywh[0][1])
    wx, wy = pixel_to_world(u, v, img_w, img_h, fovy_deg,
                             cam_x, cam_y, cam_z, table_z)

    return np.array([wx, wy, cube_z])


def update_camera_window(renderer, data, cam_name, label, target_colour,
                         window_name="FRANK — Overhead Camera"):
    """Refresh overhead camera window with stage label."""
    renderer.update_scene(data, camera=cam_name)
    frame     = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.putText(frame_bgr, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame_bgr, f"Target: {target_colour}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)