import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# test a few joint configs
configs = {
    "home":        [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853],
    "j6=1.90":     [0.0, 0.3, 0.0, -1.57079, 0.0, 1.90, 0.0],
    "j2=-0.5":     [0.0, -0.5, 0.0, -2.0, 0.0, 1.57, 0.0],
}

for name, cfg in configs.items():
    data.qpos[:7] = cfg
    mujoco.mj_forward(model, data)
    zaxis = data.xmat[hand_id].reshape(3,3)[:, 2]
    print(f"{name:12s}  zaxis={zaxis.round(3)}")