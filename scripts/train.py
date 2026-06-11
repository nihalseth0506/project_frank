import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from environment.reach_env import FrankReachEnv


# --- paths ---
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "trained"
)
LOGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


class SaveNormCallback(BaseCallback):
    # saves VecNormalize stats alongside every checkpoint
    # ensures vec_normalize.pkl is always available
    # even if training is stopped early before train() finishes
    def __init__(self, save_freq, save_path, vec_normalize_env, verbose=0):
        super().__init__(verbose)
        self.save_freq         = save_freq
        self.save_path         = save_path
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.vec_normalize_env.save(path)

        return True


def make_env():
    env = FrankReachEnv(render_mode=None)

    return Monitor(env)


def train():
    print("=" * 50)
    print("FRANK — RL Training")
    print("Task: Franka FR3 Reach")
    print("=" * 50)

    # --- create vectorized training environment ---
    train_env = make_vec_env(make_env, n_envs=1)

    # wrap with observation and reward normalization
    # scales all observations to mean=0 std=1 automatically
    train_env = VecNormalize(
        train_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0
    )

    # --- create separate evaluation environment ---
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs    = True,
        norm_reward = False,
        clip_obs    = 10.0,
        training    = False
    )

    # --- checkpoint callback ---
    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = MODELS_DIR,
        name_prefix = "frank_reach_ppo"
    )

    # --- save norm callback ---
    # saves vec_normalize.pkl every 10k steps alongside checkpoints
    save_norm_callback = SaveNormCallback(
        save_freq         = 10_000,
        save_path         = MODELS_DIR,
        vec_normalize_env = train_env
    )

    # --- eval callback ---
    # also saves vec_normalize.pkl to best/ folder when new best is found
    # ensures best_model.zip and vec_normalize.pkl are always in sync
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODELS_DIR, "best"),
        log_path             = LOGS_DIR,
        eval_freq            = 5_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
        callback_on_new_best = SaveNormCallback(
            save_freq         = 1,
            save_path         = os.path.join(MODELS_DIR, "best"),
            vec_normalize_env = train_env
        )
    )

    # --- define the PPO agent ---
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        verbose       = 1,
        learning_rate = 1e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.05,
        policy_kwargs = dict(
            net_arch = [256, 256]
        ),
        tensorboard_log = LOGS_DIR
    )

    print(f"\nPolicy network : 20 → 256 → 256 → 7")
    print(f"Training steps : 1,000,000")
    print(f"Learning rate  : 1e-4")
    print(f"Entropy coef   : 0.05")
    print(f"Obs normalized : True")
    print(f"Curriculum     : radius 0.1m → 0.4m over 500k steps\n")

    model.learn(
        total_timesteps = 1_000_000,
        callback        = [checkpoint_callback, eval_callback, save_norm_callback],
        progress_bar    = True
    )

    # --- save final policy and normalization stats ---
    final_path = os.path.join(MODELS_DIR, "frank_reach_ppo_final")
    model.save(final_path)
    train_env.save(os.path.join(MODELS_DIR, "vec_normalize.pkl"))

    print(f"\nTraining complete.")
    print(f"Policy saved to     : {final_path}")
    print(f"Norm stats saved to : {MODELS_DIR}/vec_normalize.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()