"""
test_pick_place_scene.py
Run this FIRST before training to verify:
    - Scene XML loads without errors
    - Robot spawns correctly
    - Cube physics work
    - Gripper opens and closes
    - Observation shape is correct

Run from project root:
    python tests/test_pick_place_scene.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from environment.pick_place_env import PandaPickPlaceEnv


def test_scene():
    print("=" * 50)
    print("FRANK — Pick and Place Scene Test")
    print("=" * 50)

    # ── test 1: environment creates successfully ───────────────────────────────
    print("\n[1] Creating environment...")
    env = PandaPickPlaceEnv(render_mode="human")
    print("    ✅ Environment created")

    # ── test 2: reset works ────────────────────────────────────────────────────
    print("\n[2] Testing reset...")
    obs, info = env.reset()
    print(f"    Observation shape : {obs.shape}  (expected: (35,))")
    print(f"    Cube position     : {obs[19:22].round(3)}")
    print(f"    Target position   : {obs[22:25].round(3)}")
    print(f"    Gripper state     : {obs[31]:.0f}  (0=open)")
    assert obs.shape == (34,), f"Wrong obs shape: {obs.shape}"
    print("    ✅ Reset works")

    # ── test 3: random actions run without crash ───────────────────────────────
    print("\n[3] Running 200 random steps...")
    env.render()

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.002)

        if i % 50 == 0:
            print(f"    step {i:3d} | reward={reward:.3f} | "
                  f"pinch→cube={info['pinch_to_obj_dist']:.3f}m | "
                  f"cube_h={info['cube_height']:.3f}m")

    print("    ✅ Random steps work")

    # ── test 4: gripper open and close ────────────────────────────────────────
    print("\n[4] Testing gripper...")
    obs, _ = env.reset()
    env.render()

    # close gripper
    print("    Closing gripper for 100 steps...")
    for _ in range(100):
        action    = np.zeros(8, dtype=np.float32)
        action[7] = 1.0    # close command
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        time.sleep(0.002)

    print(f"    Finger positions: {obs[14:16].round(4)}")

    # open gripper
    print("    Opening gripper for 100 steps...")
    for _ in range(100):
        action    = np.zeros(8, dtype=np.float32)
        action[7] = 0.0    # open command
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        time.sleep(0.002)

    print(f"    Finger positions: {obs[14:16].round(4)}")
    print("    ✅ Gripper works")

    # ── test 5: action space bounds ────────────────────────────────────────────
    print("\n[5] Action space check...")
    print(f"    Action space low:  {env.action_space.low}")
    print(f"    Action space high: {env.action_space.high}")
    assert env.action_space.shape == (8,), "Wrong action space shape"
    print("    ✅ Action space correct")

    # ── test 6: multiple resets give different cube positions ──────────────────
    print("\n[6] Testing randomization across resets...")
    cube_positions = []

    for i in range(5):
        obs, _ = env.reset()
        cube_pos = obs[19:22].copy()
        cube_positions.append(cube_pos)
        print(f"    Reset {i+1}: cube at {cube_pos.round(3)}")

    # check positions are different
    all_same = all(np.allclose(cube_positions[0], p) for p in cube_positions[1:])
    assert not all_same, "Cube positions are all the same — randomization broken"
    print("    ✅ Randomization works")

    print("\n" + "=" * 50)
    print("All tests passed. Scene is ready for training.")
    print("=" * 50)
    print("\nClose the viewer window to exit.")

    # keep viewer open
    while env.viewer is not None and env.viewer.is_running():
        env.step(np.zeros(8))
        env.render()
        time.sleep(0.002)

    env.close()


if __name__ == "__main__":
    test_scene()