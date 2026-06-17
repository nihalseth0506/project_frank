import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(r"models\pick_place_scene.xml")
data  = mujoco.MjData(model)

# print all body and geom names to find finger pads
print("=== BODIES ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {i}: {name}")

print("\n=== GEOMS ===")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    print(f"  {i}: {name}")

print("\n=== SITES ===")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {i}: {name}")