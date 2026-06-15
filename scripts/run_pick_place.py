import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from environment.pick_place_env import PandaPickPlaceEnv


def make_env():
    return Monitor(PandaPickPlaceEnv(render_mode="human"))


def run_policy(model_path, norm_path, n_episodes=20):
    print("=" * 50)
    print("FRANK — Pick and Place Policy Run")
    print("=" * 50)
    print(f"Loading policy    : {model_path}")
    print(f"Loading norm stats: {norm_path}")

    # create env with normalization
    env = make_vec_env(make_env, n_envs=1)
    env = VecNormalize.load(norm_path, env)
    env.training    = False
    env.norm_reward = False

    # load trained policy
    model = PPO.load(model_path, env=env)

    obs            = env.reset()
    episode        = 0
    total_success  = 0
    episode_reward = 0.0
    step_count     = 0

    print(f"\nRunning {n_episodes} episodes. Close viewer to stop early.\n")
    print(f"{'Episode':>8}  {'Result':>8}  {'Reward':>10}  "
          f"{'Steps':>6}  {'Cube→Tgt':>10}  {'Grasped':>8}")
    print("-" * 60)

    while episode < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        step_count     += 1

        if done[0]:
            real_info = info[0]
            success   = real_info.get("is_success", False)

            if success:
                total_success += 1
                result = "SUCCESS"
            else:
                result = "FAILED"

            obj_tgt_dist = real_info.get("obj_to_tgt_dist", 0)
            grasped      = real_info.get("grasped", False)

            print(f"{episode+1:>8}  {result:>8}  {episode_reward:>10.2f}  "
                  f"{step_count:>6}  {obj_tgt_dist:>10.4f}m  "
                  f"{'YES' if grasped else 'NO':>8}")

            obs            = env.reset()
            episode       += 1
            episode_reward = 0.0
            step_count     = 0

        # check if viewer was closed
        try:
            inner_env = env.envs[0].env.env
            if inner_env.viewer is not None:
                if not inner_env.viewer.is_running():
                    print("\nViewer closed — stopping.")
                    break
        except Exception:
            pass

        time.sleep(0.002)

    print("-" * 60)
    print(f"\nFinal: {total_success}/{episode} = "
          f"{total_success/max(episode,1)*100:.1f}% success rate")

    env.close()


if __name__ == "__main__":
    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pick_place_dir = os.path.join(base, "models", "pick_place")

    # find the latest trained version
    existing = [
        d for d in os.listdir(pick_place_dir)
        if d.startswith("trained_v") and
        os.path.isdir(os.path.join(pick_place_dir, d))
    ]

    if not existing:
        print("No trained models found. Run train_pick_place.py first.")
        sys.exit(1)

    versions   = [int(d.replace("trained_v", "")) for d in existing
                  if d.replace("trained_v", "").isdigit()]
    latest     = max(versions)
    version_dir = os.path.join(pick_place_dir, f"trained_v{latest}")
    # to run a specific version instead of latest:
    #version_dir = os.path.join(pick_place_dir, "trained_v1")

    model_path = os.path.join(version_dir, "best", "best_model")
    norm_path  = os.path.join(version_dir, "best", "vec_normalize.pkl")

    print(f"Using latest version: trained_v{latest}")

    if not os.path.exists(model_path + ".zip"):
        print(f"No best model found in trained_v{latest}/best/")
        print("Training may not have completed a successful eval yet.")
        sys.exit(1)

    run_policy(model_path, norm_path, n_episodes=20)