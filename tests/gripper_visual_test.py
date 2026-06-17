"""
gripper_visual_test.py
Run this to visually confirm gripper open/close convention.
Saves to tests/gripper_visual_test.py

Usage: python tests/gripper_visual_test.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import time
import numpy as np

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pick_place_scene.xml"
)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data  = mujoco.MjData(model)

# set home pose
home = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
data.qpos[:7] = home
data.ctrl[:7] = home
mujoco.mj_forward(model, data)

print("Opening viewer...")
print("Watch the gripper fingers carefully.")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Setting ctrl[7] = 0   → should be OPEN or CLOSED?")
    data.ctrl[7] = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)

    input("  → Press Enter after observing (0 = open or closed?): ")

    print("Setting ctrl[7] = 255 → should be OPEN or CLOSED?")
    data.ctrl[7] = 255.0
    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)

    input("  → Press Enter after observing (255 = open or closed?): ")

    print()
    print("Now cycling between 0 and 255 slowly...")
    for cycle in range(3):
        print(f"  Cycle {cycle+1}: setting 0...")
        data.ctrl[7] = 0.0
        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)

        print(f"  Cycle {cycle+1}: setting 255...")
        data.ctrl[7] = 255.0
        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)

    print("Done.")