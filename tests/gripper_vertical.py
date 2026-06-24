import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("models/pick_place_scene.xml")
data  = mujoco.MjData(model)

pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

# set arm to home pose
data.qpos[:7] = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
mujoco.mj_forward(model, data)

mat       = data.site_xmat[pinch_id].reshape(3, 3)
gripper_x = mat[:, 0]
gripper_y = mat[:, 1]
gripper_z = mat[:, 2]

print(f"gripper x-axis: {gripper_x.round(3)}")
print(f"gripper y-axis: {gripper_y.round(3)}")
print(f"gripper z-axis: {gripper_z.round(3)}")

world_down = np.array([0.0, 0.0, -1.0])
print(f"dot(gripper_z, world_down) = {np.dot(gripper_z, world_down):.3f}")