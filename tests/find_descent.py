import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)

lf_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
hand_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

# start from config 1 and increase joint2 to descend
base = [0.0, -0.5, 0.0, -2.0, 0.0, 1.57, 0.0]

for j2 in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    cfg = base.copy()
    cfg[1] = j2
    data.qpos[:7] = cfg
    data.ctrl[:7] = cfg
    mujoco.mj_forward(model, data)
    
    hand  = data.xpos[hand_id]
    pinch = data.site_xpos[pinch_id]
    lf    = data.xpos[lf_id]
    rf    = data.xpos[rf_id]
    zaxis = data.xmat[hand_id].reshape(3,3)[:,2]
    
    # finger geom positions
    geom_z = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] in [lf_id, rf_id]:
            geom_z.append(data.geom_xpos[i][2])
    
    print(f"j2={j2:5.2f}  hand_z={hand[2]:.3f}  "
          f"pinch_z={pinch[2]:.3f}  "
          f"finger_z={lf[2]:.3f}  "
          f"geom_z_min={min(geom_z):.3f}  "
          f"hand_xy=[{hand[0]:.3f},{hand[1]:.3f}]  "
          f"zaxis=[{zaxis[0]:.2f},{zaxis[2]:.2f}]")