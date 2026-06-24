"""
reach_env_panda.py

Reach environment for Franka Panda arm — uses pinch site as EE reference.
v8 — top-down orientation reward added, joint constraints removed from step().
The policy learns to approach targets from above rather than from the side.

Observation (20 values):
    7  joint positions
    7  joint velocities
    3  pinch site xyz
    3  target xyz

Action (7 values):
    7  joint deltas (±DELTA_LIMIT radians)
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")

HOME_POSE = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])

# curriculum centered above table at reachable top-down height
CURRICULUM_CENTER       = np.array([0.50, 0.0, 0.60])
CURRICULUM_START_RADIUS = 0.08
CURRICULUM_END_RADIUS   = 0.35

TARGET_LOW  = np.array([0.25, -0.40, 0.05])
TARGET_HIGH = np.array([0.75,  0.40, 0.75])

CURRICULUM_STEPS  = 500_000
MAX_EPISODE_STEPS = 1000
GOAL_THRESHOLD    = 0.04
DELTA_LIMIT       = 0.05

# orientation reward weight — how much top-down alignment is valued
ORIENTATION_WEIGHT = 0.2


class PandaReachEnv(gym.Env):
    """
    Reach environment for Franka Panda.
    Top-down variant — policy learns vertical gripper orientation via reward.
    No hard joint constraints in step() — orientation emerges from training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        self._pinch_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch"
        )

        self.current_step = 0
        self.total_steps  = 0
        self.target_pos   = np.zeros(3)

        self.action_space = spaces.Box(
            low   = -DELTA_LIMIT * np.ones(7, dtype=np.float32),
            high  =  DELTA_LIMIT * np.ones(7, dtype=np.float32),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

    def _get_ee_pos(self):
        return self.data.site_xpos[self._pinch_id].copy()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:7].copy(),
            self.data.qvel[:7].copy(),
            self._get_ee_pos(),
            self.target_pos
        ]).astype(np.float32)

    def _get_distance(self):
        return float(np.linalg.norm(self._get_ee_pos() - self.target_pos))

    def _get_orientation_reward(self):
        """
        Reward for gripper pointing straight down.
        Reads the pinch site rotation matrix — z-axis of gripper frame
        should align with world -z (pointing down).
        Returns value in [-1, 1]: 1.0 = perfect top-down, -1.0 = upside-down.
        """
        pinch_mat  = self.data.site_xmat[self._pinch_id].reshape(3, 3)
        gripper_z  = pinch_mat[:, 2]
        world_down = np.array([0.0, 0.0, -1.0])
        return float(np.dot(gripper_z, world_down))

    def _compute_reward(self, distance):
        reward = -distance * 2.0

        orientation_score = self._get_orientation_reward()

        # small constant orientation nudge every step
        reward += 0.2 * orientation_score

        # big bonuses only unlocked when gripper is pointing down
        if orientation_score > 0.7:
            if distance < 0.20: reward += 0.5
            if distance < 0.10: reward += 2.0
            if distance < 0.05: reward += 5.0
            if distance < GOAL_THRESHOLD: reward += 10.0

        return reward

    def _get_curriculum_radius(self):
        progress = min(self.total_steps / CURRICULUM_STEPS, 1.0)
        return (CURRICULUM_START_RADIUS +
                progress * (CURRICULUM_END_RADIUS - CURRICULUM_START_RADIUS))

    def _move_to_home(self):
        self.data.qpos[:7] = HOME_POSE.copy()
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = HOME_POSE.copy()
        self.data.ctrl[7]  = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._move_to_home()

        radius = self._get_curriculum_radius()
        low    = np.maximum(CURRICULUM_CENTER - radius, TARGET_LOW)
        high   = np.minimum(CURRICULUM_CENTER + radius, TARGET_HIGH)

        self.target_pos   = np.random.uniform(low, high).astype(np.float32)
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        current = self.data.ctrl[:7].copy()
        new_arm = current + action.astype(np.float64)

        # no orientation constraints — policy learns to orient top-down from reward
        for i in range(7):
            lo = self.model.actuator_ctrlrange[i, 0]
            hi = self.model.actuator_ctrlrange[i, 1]
            new_arm[i] = np.clip(new_arm[i], lo, hi)
        
        # in step() — restore these lines
        new_arm[5] = np.clip(new_arm[5], 1.70, 2.10)
        new_arm[6] = np.clip(new_arm[6], -0.2, 0.2)

        self.data.ctrl[:7] = new_arm
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        self.total_steps  += 1

        obs        = self._get_obs()
        distance   = self._get_distance()
        reward     = self._compute_reward(distance)
        terminated = bool(distance < GOAL_THRESHOLD)
        truncated  = bool(self.current_step >= MAX_EPISODE_STEPS)

        info = {
            "distance":    distance,
            "is_success":  terminated,
            "orientation": self._get_orientation_reward()
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None