import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

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


def make_env():
    # creates one training environment instance
    # Monitor wrapper logs episode rewards and lengths automatically
    env = FrankReachEnv(render_mode=None)

    return Monitor(env)


def train():
    print("=" * 50)
    print("FRANK — RL Training")
    print("Task: Franka FR3 Reach")
    print("=" * 50)

    # --- create vectorized training environment ---
    # n_envs=1 for now — your laptop runs one env comfortably
    # later we can increase this for faster training
    train_env = make_vec_env(make_env, n_envs=1)

    # --- create separate evaluation environment ---
    # we evaluate on a separate env so training isn't interrupted
    eval_env = Monitor(FrankReachEnv(render_mode=None))

    # --- checkpoint callback ---
    # saves policy weights every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq  = 10_000,
        save_path  = MODELS_DIR,
        name_prefix = "frank_reach_ppo"
    )

    # --- eval callback ---
    # runs 10 evaluation episodes every 5000 steps
    # saves the best policy found so far
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODELS_DIR, "best"),
        log_path             = LOGS_DIR,
        eval_freq            = 5_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False
    )

    # --- define the PPO agent ---
    model = PPO(
        policy  = "MlpPolicy",   # multi-layer perceptron — standard for state-based RL
        env     = train_env,
        verbose = 1,             # print training progress

        # --- core PPO hyperparameters ---
        learning_rate = 3e-4,    # how fast weights update — 0.0003
        n_steps       = 2048,    # steps collected before each weight update
        batch_size    = 64,      # samples per gradient update
        n_epochs      = 10,      # passes through collected data per update
        gamma         = 0.99,    # discount factor — how much future rewards matter
        gae_lambda    = 0.95,    # advantage estimation smoothing

        # entropy coefficient — forces agent to keep exploring
        # prevents the collapse we saw in v1
        ent_coef      = 0.01,

        # --- network architecture ---
        policy_kwargs = dict(
            net_arch = [256, 256]  # two hidden layers of 256 neurons each
        ),

        tensorboard_log = LOGS_DIR
    )

    print(f"\nPolicy network architecture: 20 → 256 → 256 → 7")
    print(f"Total training timesteps   : 300,000")
    print(f"Checkpoints saved to       : {MODELS_DIR}")
    print(f"Logs saved to              : {LOGS_DIR}")
    print(f"\nStarting training...\n")

    # --- train ---
    model.learn(
        total_timesteps    = 500_000,
        callback           = [checkpoint_callback, eval_callback],
        progress_bar       = True
    )

    # --- save final policy ---
    final_path = os.path.join(MODELS_DIR, "frank_reach_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete. Final policy saved to: {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()