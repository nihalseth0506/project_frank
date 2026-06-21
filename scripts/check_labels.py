import cv2
import os
import numpy as np

# check 5 random samples
img_dir = r"data\images\train"
lbl_dir = r"data\labels\train"

files = os.listdir(img_dir)[:5]

for fname in files:
    img_path = os.path.join(img_dir, fname)
    lbl_path = os.path.join(lbl_dir, fname.replace(".png", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(lbl_path) as f:
        line = f.readline().strip().split()

    cls, cx, cy, bw, bh = [float(x) for x in line]

    # convert YOLO normalized to pixel
    px = int(float(cx) * w)
    py = int(float(cy) * h)
    pw = int(float(bw) * w)
    ph = int(float(bh) * h)

    # draw box and center dot
    cv2.rectangle(img,
                  (px - pw//2, py - ph//2),
                  (px + pw//2, py + ph//2),
                  (0, 255, 0), 2)
    cv2.circle(img, (px, py), 4, (0, 0, 255), -1)

    cv2.imwrite(f"check_{fname}", img)
    print(f"{fname}: cube center pixel=({px},{py})")