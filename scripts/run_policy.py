import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from environment.reach_env import FrankReachEnv


def make_env():
    return Monitor(FrankReachEnv(render_mode="human"))


def run_policy(model_path, norm_path):
    print(f"Loading policy: {model_path}")

    # create env without normalization wrapper
    # norm stats were not saved — running without them
    env = make_vec_env(make_env, n_envs=1)

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    episode        = 0
    total_success  = 0
    episode_reward = 0.0

    print(f"\nRunning 20 episodes. Close viewer to stop early.\n")

    while episode < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]

        if done[0]:
            real_info = info[0]

            if real_info.get("is_success", False):
                total_success += 1
                result = "SUCCESS"
            else:
                result = "FAILED"

            print(f"Episode {episode+1:2d} | {result} | "
                  f"reward: {episode_reward:.2f} | "
                  f"distance: {real_info.get('distance', 0):.4f}m")

            obs            = env.reset()
            episode       += 1
            episode_reward = 0.0

    print(f"\nFinal result: {total_success}/20 = "
          f"{total_success/20*100:.1f}% success rate")

    env.close()


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(
        base, "models", "trained_v4_normalized", "best", "best_model"
    )

    # norm_path not used — running without normalization
    run_policy(model_path, norm_path=None)