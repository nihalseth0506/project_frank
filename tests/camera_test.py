import mujoco
import numpy as np
import cv2

model    = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data     = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="overhead_cam")
frame = renderer.render()

cv2.imwrite("test_overhead.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print("Saved test_overhead.png")