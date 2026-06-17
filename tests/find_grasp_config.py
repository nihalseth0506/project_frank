import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)

lf_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

# test joint configurations that might give top-down grasp
configs = [
    [0.0, -0.5, 0.0, -2.0, 0.0, 1.57, 0.0],    # elbow up config
    [0.0,  0.5, 0.0, -1.5, 0.0, 1.57, 0.0],    # elbow down config
    [0.0,  1.0, 0.0, -2.5, 0.0, 1.57, 0.0],    # more elbow
    [0.0,  0.8, 0.0, -2.0, 0.0, 1.57, 0.785],  # rotated wrist
]

for cfg in configs:
    data.qpos[:7] = cfg
    data.ctrl[:7] = cfg
    mujoco.mj_forward(model, data)
    
    hand  = data.xpos[hand_id]
    pinch = data.site_xpos[pinch_id]
    lf    = data.xpos[lf_id]
    
    # hand z-axis (pointing direction)
    hand_zaxis = data.xmat[hand_id].reshape(3,3)[:,2]
    
    print(f"joints={[round(x,2) for x in cfg]}")
    print(f"  hand pos={hand.round(3)}  pinch={pinch.round(3)}")
    print(f"  finger z={lf[2]:.3f}")
    print(f"  hand z-axis={hand_zaxis.round(3)}")
    print()