import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot.controller    import RobotController
from robot.observer      import RobotObserver
from config.robot_config import (
    MODEL_PATH,
    TIMESTEP,
    MAX_EPISODE_STEPS,
    GOAL_THRESHOLD,
    REWARD_SCALE,
    TARGET_LOW,
    TARGET_HIGH,
    HOME_POSE,
    END_EFFECTOR_BODY,
    DEFAULT_MOVE_STEPS
)


class FrankReachEnv(gym.Env):
    # tells gymnasium this environment supports rgb rendering
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode  = render_mode
        self.render_mode  = render_mode

        # load model and data directly here
        # we dont use RobotController.run() in RL
        # we need manual control of every step
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)

        # observer handles all state reading
        self.observer = RobotObserver(self.model, self.data)

        # viewer is only created if render_mode is human
        self.viewer = None

        # track steps within current episode
        self.current_step = 0

        # target position in 3D space — randomized each episode
        self.target_pos = np.zeros(3)

        # delta action space — small changes per step
        # limits how much each joint can move per step
        # prevents violent jumpy motion
        DELTA_LIMIT = 0.05  # radians per step maximum

        self.action_space = spaces.Box(
            low   = -DELTA_LIMIT * np.ones(7, dtype=np.float32),
            high  =  DELTA_LIMIT * np.ones(7, dtype=np.float32),
            dtype = np.float32
        )

        # --- define observation space ---
        # what the agent sees each step:
        # 7 joint positions + 7 joint velocities + 3 end effector pos
        # + 3 target position = 20 numbers total
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (20,),
            dtype = np.float32
        )

    def _get_obs(self):
        # build the 20-dimensional observation vector
        obs_dict = self.observer.get_observation(END_EFFECTOR_BODY)

        return np.concatenate([
            obs_dict["joint_positions"],    # 7 numbers
            obs_dict["joint_velocities"],   # 7 numbers
            obs_dict["end_effector_pos"],   # 3 numbers
            self.target_pos                 # 3 numbers
        ]).astype(np.float32)              # total = 20

    def _get_distance(self):
        # euclidean distance between end effector and target
        ee_pos = self.observer.get_end_effector_pos(END_EFFECTOR_BODY)

        return np.linalg.norm(ee_pos - self.target_pos)

    def _compute_reward(self, distance):
        # dense distance reward
        reward = -distance * REWARD_SCALE

        # bonus for getting very close — encourages final precision
        if distance < 0.1:
            reward += 1.0

        # big bonus for success — encourages actually reaching the goal
        if distance < GOAL_THRESHOLD:
            reward += 10.0

        return reward

    def reset(self, seed=None, options=None):
        # required by gymnasium — handles random seed
        super().reset(seed=seed)

        # reset simulation to initial state
        mujoco.mj_resetData(self.model, self.data)

        # move to home pose so robot starts consistently
        self._move_to_home()

        # spawn target at a random position within bounds
        self.target_pos = np.random.uniform(
            low  = TARGET_LOW,
            high = TARGET_HIGH
        )

        # reset step counter for this episode
        self.current_step = 0

        obs  = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # action is now a delta — add to current joint positions
        current_angles = self.data.qpos[:7].copy()
        new_angles     = current_angles + action

        # clip to joint limits so delta never exceeds physical bounds
        for i in range(7):
            low  = self.model.jnt_range[i][0]
            high = self.model.jnt_range[i][1]
            new_angles[i] = np.clip(new_angles[i], low, high)

        # apply clipped angles
        self.data.ctrl[:7] = new_angles

        # advance physics
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # get observation and compute reward
        obs      = self._get_obs()
        distance = self._get_distance()
        reward   = self._compute_reward(distance)

        terminated = bool(distance < GOAL_THRESHOLD)
        truncated  = bool(self.current_step >= MAX_EPISODE_STEPS)

        info = {
            "distance"  : distance,
            "target_pos": self.target_pos,
            "is_success": terminated
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _move_to_home(self):
        # move robot to home pose without rendering
        # used during reset to ensure consistent start state
        start = self.data.qpos[:7].copy()
        target = np.array(HOME_POSE)

        for step in range(DEFAULT_MOVE_STEPS):
            t             = step / DEFAULT_MOVE_STEPS
            interpolated  = start + t * (target - start)
            self.data.ctrl[:7] = interpolated
            mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None