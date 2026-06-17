"""
Run this to find exact finger pad positions during grasp position.
Save to tests/finger_pos_test.py and run:
    python tests/finger_pos_test.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pick_place_scene.xml"
)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data  = mujoco.MjData(model)

# set arm to approximate grasp position (from logs: joints near home but descended)
# use values close to what we see during descent
grasp_joints = np.array([0.0, 0.8, 0.3, -2.42, 1.22, 2.1, 0.2])
data.qpos[:7] = grasp_joints
data.ctrl[:7] = grasp_joints
data.ctrl[7]  = 0.0  # closed
mujoco.mj_forward(model, data)

pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
lf_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
hand_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

pinch = data.site_xpos[pinch_id]
lf    = data.xpos[lf_id]
rf    = data.xpos[rf_id]
hand  = data.xpos[hand_id]

print(f"Hand body pos    : {hand.round(4)}")
print(f"Left finger pos  : {lf.round(4)}")
print(f"Right finger pos : {rf.round(4)}")
print(f"Pinch site pos   : {pinch.round(4)}")
print()
print(f"Pinch z          : {pinch[2]:.4f}")
print(f"Left finger z    : {lf[2]:.4f}")
print(f"Right finger z   : {rf[2]:.4f}")
print(f"Hand z           : {hand[2]:.4f}")
print()

# finger orientation
lf_xmat = data.xmat[lf_id].reshape(3,3)
print(f"Left finger rotation matrix:")
print(lf_xmat.round(4))
print()

# where are finger tips (assuming 8cm long fingers)
# finger tip = finger body pos + local tip offset in world frame
FINGER_LEN = 0.06  # approximate finger length from body center to tip
lf_tip = lf + lf_xmat @ np.array([0, 0, FINGER_LEN])
rf_xmat = data.xmat[rf_id].reshape(3,3)
rf_tip = rf + rf_xmat @ np.array([0, 0, FINGER_LEN])

print(f"Estimated left finger tip  : {lf_tip.round(4)}")
print(f"Estimated right finger tip : {rf_tip.round(4)}")
print()
print(f"Cube resting z             : 0.4250")
print(f"Table surface z            : 0.4000")
print()
print(f"Left finger tip z vs cube  : {lf_tip[2]:.4f} vs 0.4250")
print(f"Gap (tip - cube center)    : {lf_tip[2] - 0.425:.4f}m")

# also print geom positions for robot geoms
print("\n=== Finger geom positions ===")
for i in range(model.ngeom):
    body_id = model.geom_bodyid[i]
    if body_id in [lf_id, rf_id]:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        pos  = data.geom_xpos[i]
        print(f"  geom {i} ({name}) body={body_id}: pos={pos.round(4)}")