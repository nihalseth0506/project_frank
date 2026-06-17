import mujoco
import numpy as np
import os

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)

home = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
data.qpos[:7] = home
data.ctrl[:7] = home
mujoco.mj_forward(model, data)

pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
print(f"Pinch at home: {data.site_xpos[pinch_id].round(4)}")