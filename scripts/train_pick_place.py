import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU — faster for MuJoCo

import sys
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

from environment.pick_place_env import PandaPickPlaceEnv


PROJ_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICK_PLACE_DIR = os.path.join(PROJ_ROOT, "models", "pick_place")


def get_next_version():
    # scan existing trained_vN folders and return next version number
    # if trained_v1 and trained_v2 exist, returns 3
    existing = [
        d for d in os.listdir(PICK_PLACE_DIR)
        if d.startswith("trained_v") and
        os.path.isdir(os.path.join(PICK_PLACE_DIR, d))
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
    return Monitor(PandaPickPlaceEnv(render_mode=None))


def train():
    # auto-detect next version number
    version    = get_next_version()
    MODELS_DIR = os.path.join(PICK_PLACE_DIR, f"trained_v{version}")
    LOGS_DIR   = os.path.join(PROJ_ROOT, "logs", "pick_place", f"v{version}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 55)
    print("FRANK — Pick and Place Training")
    print("Robot: Franka Panda + Parallel Jaw Gripper")
    print("=" * 55)
    print(f"Version    : v{version}")
    print(f"Models dir : {MODELS_DIR}")
    print(f"Logs dir   : {LOGS_DIR}")

    # ── environments ───────────────────────────────────────────────────────────
    train_env = make_vec_env(make_env, n_envs=1)
    train_env = VecNormalize(
        train_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0
    )

    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        norm_obs    = True,
        norm_reward = False,
        clip_obs    = 10.0,
        training    = False
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = MODELS_DIR,
        name_prefix = "pick_place_ppo"
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

    # ── PPO model ──────────────────────────────────────────────────────────────
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        verbose       = 1,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.02,
        clip_range    = 0.2,
        policy_kwargs = dict(
            net_arch = [512, 512]
        ),
        tensorboard_log = LOGS_DIR
    )

    print(f"\nPolicy network : 34 → 512 → 512 → 8")
    print(f"Training steps : 5,000,000")
    print(f"Learning rate  : 3e-4")
    print(f"Entropy coef   : 0.02\n")

    # ── train ──────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = 5_000_000,
        callback        = [checkpoint_callback, eval_callback, save_norm_callback],
        progress_bar    = True
    )

    # ── save final ─────────────────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "pick_place_ppo_final")
    model.save(final_path)
    train_env.save(os.path.join(MODELS_DIR, "vec_normalize.pkl"))

    print(f"\nTraining complete — v{version}")
    print(f"Policy    : {final_path}")
    print(f"Norm stats: {MODELS_DIR}/vec_normalize.pkl")
    print(f"Best model: {MODELS_DIR}/best/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()