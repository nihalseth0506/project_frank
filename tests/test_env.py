import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.reach_env import FrankReachEnv


def test_environment():
    print("Creating environment...")
    env = FrankReachEnv(render_mode="human")

    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space.low[:3]} to "
          f"{env.action_space.high[:3]}")
    print(f"Target position   : {env.target_pos}")

    print("\nRunning with random actions. Close window to stop.")

    episode      = 0
    total_steps  = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        env.render()

        if total_steps % 200 == 0:
            print(f"Episode {episode:3d} | "
                  f"step {env.current_step:4d} | "
                  f"reward: {reward:.4f} | "
                  f"distance: {info['distance']:.4f}m | "
                  f"success: {info['is_success']}")

        if terminated or truncated:
            episode += 1
            obs, info = env.reset()

        # stop if viewer is closed
        if env.viewer is not None and not env.viewer.is_running():
            break

    env.close()
    print(f"\nDone. Ran {episode} episodes, {total_steps} total steps.")


if __name__ == "__main__":
    test_environment()