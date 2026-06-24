"""
train_reach_panda.py

Train PPO reach policy on Franka Panda arm.
v8 — top-down orientation reward, no joint constraints, expanded curriculum.

Saves to models/reach/panda/trained_vN/ with auto-versioning.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU faster for MuJoCo

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from environment.reach_env_panda import PandaReachEnv


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDA_DIR = os.path.join(PROJ_ROOT, "models", "reach", "panda")
os.makedirs(PANDA_DIR, exist_ok=True)


def get_next_version():
    existing = [
        d for d in os.listdir(PANDA_DIR)
        if d.startswith("trained_v") and
        os.path.isdir(os.path.join(PANDA_DIR, d))
    ]
    if not existing:
        return 1
    versions = []
    for d in existing:
        try:
            versions.append(int(d.replace("trained_v", "")))
        except ValueError:
            pass
    return max(versions) + 1 if versions else 1


class SaveNormCallback(BaseCallback):
    """Save VecNormalize stats alongside every checkpoint and best model."""

    def __init__(self, save_freq, save_path, vec_normalize_env, verbose=0):
        super().__init__(verbose)
        self.save_freq         = save_freq
        self.save_path         = save_path
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.vec_normalize_env.save(
                os.path.join(self.save_path, "vec_normalize.pkl")
            )
        return True


def make_env():
    return Monitor(PandaReachEnv(render_mode=None))


def train():
    version    = get_next_version()
    MODELS_DIR = os.path.join(PANDA_DIR, f"trained_v{version}")
    LOGS_DIR   = os.path.join(PROJ_ROOT, "logs", "reach_panda", f"v{version}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 55)
    print("FRANK — Panda Reach Training (Top-Down v8)")
    print("Orientation reward — no hard joint constraints")
    print("=" * 55)
    print(f"Version          : v{version}")
    print(f"Models dir       : {MODELS_DIR}")
    print(f"Curriculum center: [0.50, 0.0, 0.60]")
    print(f"Curriculum radius: 0.08m → 0.35m over 500k steps")
    print(f"Orientation weight: 1.0 per step")

    train_env = make_vec_env(make_env, n_envs=1)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = MODELS_DIR,
        name_prefix = "panda_reach_ppo"
    )

    save_norm_callback = SaveNormCallback(
        save_freq         = 10_000,
        save_path         = MODELS_DIR,
        vec_normalize_env = train_env
    )

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

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        verbose         = 1,
        learning_rate   = 1e-4,
        n_steps         = 4096,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        ent_coef        = 0.05,
        policy_kwargs   = dict(net_arch=[256, 256]),
        tensorboard_log = LOGS_DIR
    )

    print(f"\nPolicy network : 20 → 256 → 256 → 7")
    print(f"Training steps : 2,000,000")
    print(f"Learning rate  : 1e-4")
    print(f"Ent coef       : 0.05\n")

    model.learn(
        total_timesteps = 2_000_000,
        callback        = [checkpoint_callback, eval_callback, save_norm_callback],
        progress_bar    = True
    )

    final_path = os.path.join(MODELS_DIR, "panda_reach_ppo_final")
    model.save(final_path)
    train_env.save(os.path.join(MODELS_DIR, "vec_normalize.pkl"))

    print(f"\nTraining complete — v{version}")
    print(f"Best model : {MODELS_DIR}/best/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()