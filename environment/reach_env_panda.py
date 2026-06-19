"""
reach_env_panda.py

Reach environment for Franka Panda arm — uses pinch site as EE reference.

The pinch site sits at [0, 0, 0.175] from the hand body — fingertip position.
This matches pick_place_scene.xml which also defines the pinch site.
Training with pinch site means the policy learns to put the fingertip
at the target, not the wrist — fixing the wrist-pointing-up problem.
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

# use the pick_place_scene.xml — it has the pinch site already defined
# and uses panda.xml internally
SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")

# Panda home pose (7 arm joints)
HOME_POSE = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])

# curriculum centered on pinch site position at home pose
# pinch site at home ≈ [0.557, 0.0, 0.391] (hand [0.664,0,0.481] + offset)
# we set curriculum slightly higher and forward for table-level tasks
CURRICULUM_CENTER       = np.array([0.45, 0.0, 0.38])
CURRICULUM_START_RADIUS = 0.08
CURRICULUM_END_RADIUS   = 0.30

TARGET_LOW  = np.array([0.25, -0.40, 0.05])
TARGET_HIGH = np.array([0.75,  0.40, 0.65])

CURRICULUM_STEPS        = 500_000
MAX_EPISODE_STEPS = 1000
GOAL_THRESHOLD    = 0.04
DELTA_LIMIT       = 0.05


class PandaReachEnv(gym.Env):
    """
    Reach environment for Franka Panda.

    Uses pinch site as end effector reference — matches pick_place_scene.xml.
    Policy trained here can be used directly in pick_place_scripted_env.py
    without any coordinate mismatch.

    Observation (20 values):
        7  arm joint positions
        7  arm joint velocities
        3  pinch site xyz (fingertip position)
        3  target xyz

    Action (7 values):
        7  arm joint deltas (±DELTA_LIMIT radians)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        # use pinch site — fingertip position
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
        """Pinch site position — fingertip in world frame."""
        return self.data.site_xpos[self._pinch_id].copy()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:7].copy(),   # [0:7]   arm joints
            self.data.qvel[:7].copy(),   # [7:14]  arm velocities
            self._get_ee_pos(),          # [14:17] pinch site xyz
            self.target_pos              # [17:20] target xyz
        ]).astype(np.float32)

    def _get_distance(self):
        return float(np.linalg.norm(self._get_ee_pos() - self.target_pos))

    def _compute_reward(self, distance):
        reward = -distance * 2.0

        # distance bonuses
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
        self.data.ctrl[7]  = 0.0   # gripper open
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

        # constrain joint6 to keep gripper vertical during training
        new_arm[5] = np.clip(new_arm[5], 1.70, 2.10)
        new_arm[6] = np.clip(new_arm[6], -0.2, 0.2)

        for i in range(7):
            lo = self.model.actuator_ctrlrange[i, 0]
            hi = self.model.actuator_ctrlrange[i, 1]
            new_arm[i] = np.clip(new_arm[i], lo, hi)

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

        info = {"distance": distance, "is_success": terminated}

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