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
    print(f"Loading policy    : {model_path}")
    print(f"Loading norm stats: {norm_path}")

    env = make_vec_env(make_env, n_envs=1)
    env = VecNormalize.load(norm_path, env)
    env.training    = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    obs            = env.reset()
    episode        = 0
    total_success  = 0
    episode_reward = 0.0

    print(f"\nRunning 20 episodes.\n")

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

        if env.envs[0].env.env.viewer is not None:
            if not env.envs[0].env.env.viewer.is_running():
                break

    print(f"\nFinal: {total_success}/20 = "
          f"{total_success/20*100:.1f}% success rate")

    env.close()

def test_specific_target(model_path, norm_path, target):
    print(f"\nTesting specific target: {target}")

    env = make_vec_env(make_env, n_envs=1)
    env = VecNormalize.load(norm_path, env)
    env.training    = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    # reset and manually override target position
    obs = env.reset()
    env.envs[0].env.env.target_pos = np.array(target, dtype=np.float32)

    # rebuild obs with new target
    obs = env.envs[0].env.env._get_obs()
    obs = env.normalize_obs(obs.reshape(1, -1))

    episode_reward = 0.0
    done           = False

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]

        if done[0]:
            real_info = info[0]
            result    = "SUCCESS" if real_info.get("is_success") else "FAILED"
            print(f"Result   : {result}")
            print(f"Distance : {real_info['distance']:.4f}m")
            print(f"Reward   : {episode_reward:.2f}")
            print(f"Steps    : {step+1}")

            break

    env.close()

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(
        base, "models", "trained", "best", "best_model"
    )
    norm_path = os.path.join(
        base, "models", "trained", "best", "vec_normalize.pkl"
    )

    # test three specific targets
    test_specific_target(model_path, norm_path, [0.35, 0.0,  0.45])   # center
    test_specific_target(model_path, norm_path, [0.45, 0.15, 0.50])   # right and up
    test_specific_target(model_path, norm_path, [0.25, -0.1, 0.35])   # left and down
    
    run_policy(model_path, norm_path)