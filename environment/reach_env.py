import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
    DEFAULT_MOVE_STEPS,
    CURRICULUM_START_RADIUS,
    CURRICULUM_END_RADIUS,
    CURRICULUM_STEPS
)


class FrankReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model    = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data     = mujoco.MjData(self.model)
        self.observer = RobotObserver(self.model, self.data)
        self.viewer   = None

        self.current_step = 0
        self.total_steps  = 0    # tracks steps across all episodes for curriculum
        self.target_pos   = np.zeros(3)

        # delta action space — small changes per step
        DELTA_LIMIT = 0.1   # max radians per joint per step

        self.action_space = spaces.Box(
            low   = -DELTA_LIMIT * np.ones(7, dtype=np.float32),
            high  =  DELTA_LIMIT * np.ones(7, dtype=np.float32),
            dtype = np.float32
        )

        # observation: 7 joint pos + 7 joint vel + 3 ee pos + 3 target = 20
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (20,),
            dtype = np.float32
        )

    def _get_curriculum_radius(self):
        # linearly increase difficulty as training progresses
        # at total_steps=0          → radius = 0.1m  (easy)
        # at total_steps=500k       → radius = 0.4m  (full workspace)
        # at total_steps=500k+      → radius = 0.4m  (stays at max)
        progress = min(self.total_steps / CURRICULUM_STEPS, 1.0)
        radius   = (CURRICULUM_START_RADIUS +
                    progress * (CURRICULUM_END_RADIUS - CURRICULUM_START_RADIUS))

        return radius

    def _get_obs(self):
        obs_dict = self.observer.get_observation(END_EFFECTOR_BODY)

        return np.concatenate([
            obs_dict["joint_positions"],   # [0:7]   joint angles
            obs_dict["joint_velocities"],  # [7:14]  joint velocities
            obs_dict["end_effector_pos"],  # [14:17] ee xyz in world frame
            self.target_pos                # [17:20] target xyz in world frame
        ]).astype(np.float32)

    def _get_distance(self):
        ee_pos = self.observer.get_end_effector_pos(END_EFFECTOR_BODY)

        return float(np.linalg.norm(ee_pos - self.target_pos))

    def _compute_reward(self, distance):
        # dense reward — negative distance gives gradient every step
        reward = -distance

        # bonus for getting close — encourages precision
        if distance < 0.1:
            reward += 1.0

        # large bonus for success — clearly marks the goal
        if distance < GOAL_THRESHOLD:
            reward += 10.0

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._move_to_home()

        # get home end effector position as curriculum center
        home_ee = self.observer.get_end_effector_pos(END_EFFECTOR_BODY)
        radius  = self._get_curriculum_radius()

        # sample target within current curriculum radius
        # retry until target is within safe reachable bounds
        while True:
            offset = np.random.uniform(-radius, radius, size=3)
            target = home_ee + offset

            if (TARGET_LOW[0]  <= target[0] <= TARGET_HIGH[0] and
                TARGET_LOW[1]  <= target[1] <= TARGET_HIGH[1] and
                TARGET_LOW[2]  <= target[2] <= TARGET_HIGH[2]):
                break

        self.target_pos   = target
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        # apply delta to current joint positions
        current_angles = self.data.qpos[:7].copy()
        new_angles     = current_angles + action

        # clip to joint limits
        for i in range(7):
            low          = self.model.jnt_range[i][0]
            high         = self.model.jnt_range[i][1]
            new_angles[i] = np.clip(new_angles[i], low, high)

        self.data.ctrl[:7] = new_angles
        mujoco.mj_step(self.model, self.data)

        # increment both step counters
        self.current_step += 1
        self.total_steps  += 1

        obs      = self._get_obs()
        distance = self._get_distance()
        reward   = self._compute_reward(distance)

        terminated = bool(distance < GOAL_THRESHOLD)
        truncated  = bool(self.current_step >= MAX_EPISODE_STEPS)

        info = {
            "distance"          : distance,
            "target_pos"        : self.target_pos,
            "is_success"        : terminated,
            "curriculum_radius" : self._get_curriculum_radius()
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _move_to_home(self):
        start  = self.data.qpos[:7].copy()
        target = np.array(HOME_POSE)

        for step in range(DEFAULT_MOVE_STEPS):
            t              = step / DEFAULT_MOVE_STEPS
            interpolated   = start + t * (target - start)
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