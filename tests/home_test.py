import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)

# try different joint2 values — negative pulls arm up and back
# swing arm to the side and up — out of camera view
test_pose = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853, 0.04, 0.04])
data.qpos[:9] = test_pose
data.ctrl[:7] = test_pose[:7]
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="overhead_cam")
frame = renderer.render()

import cv2
cv2.imwrite("retracted_test.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print("Saved retracted_test.png")