import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── model path ────────────────────────────────────────────────────────────────
import os
_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")

# ── task constants ─────────────────────────────────────────────────────────────
TABLE_Z     = 0.530
CUBE_HALF_H = 0.025
CUBE_Z      = TABLE_Z + CUBE_HALF_H   # 0.555

# object spawn zone (on table, in front of robot)
OBJECT_LOW  = np.array([0.30, -0.10, CUBE_Z])
OBJECT_HIGH = np.array([0.55,  0.20, CUBE_Z])

# target zone (different region on table)
TARGET_LOW  = np.array([0.30, -0.30, CUBE_Z])
TARGET_HIGH = np.array([0.55, -0.15, CUBE_Z])

# home joint configuration (7 arm joints + 2 finger joints)
HOME_QPOS = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.0])

# delta limits per step
ARM_DELTA    = 0.02    # radians per step for arm joints
GRIPPER_CTRL = 255.0   # max gripper ctrl value (fully closed)

# success thresholds
GRASP_HEIGHT      = TABLE_Z + CUBE_HALF_H + 0.04   # cube must be lifted 4cm above table
PLACE_THRESHOLD   = 0.04    # meters — how close counts as placed
GRASP_THRESHOLD   = 0.06    # meters — pinch to cube distance to count as grasped

# episode limits
MAX_EPISODE_STEPS = 3000    # longer episodes needed for pick + place


class PandaPickPlaceEnv(gym.Env):
    """
    Pick and place environment for Franka Panda arm with parallel jaw gripper.

    Task:
        1. Move gripper to red cube
        2. Close gripper and grasp cube
        3. Lift cube above table
        4. Carry cube to green target zone
        5. Place cube within target zone and open gripper

    Observation (35 values):
        7  arm joint positions
        7  arm joint velocities
        2  finger joint positions
        3  gripper pinch site xyz (end effector position)
        3  cube xyz position
        3  target xyz position
        3  vector from pinch to cube (error signal for grasping)
        3  vector from cube to target (error signal for placing)
        1  gripper open/close state (0=open, 1=closed)
        1  cube grasped flag (0/1)
        1  cube height above table (progress signal)

    Action (8 values):
        7  arm joint deltas (±ARM_DELTA radians)
        1  gripper command (0=open, 1=close — binary, agent decides)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # load model
        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        # cache body and site ids for fast lookup
        self._pinch_site_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")

        # episode state
        self.current_step  = 0
        self.target_pos    = np.zeros(3)
        self.gripper_state = 0.0    # 0=open, 1=closed
        self.phase         = 0      # 0=reach, 1=grasp, 2=lift, 3=place
        self.max_cube_height = 0.0   # tracks highest point cube reached this episode
        self.has_grasped = False   # tracks if cube was grasped this episode
        self.has_lifted  = False   # tracks if cube was lifted this episode

        # ── action space ──────────────────────────────────────────────────────
        # 7 arm deltas + 1 gripper binary command
        arm_low  = -ARM_DELTA * np.ones(7, dtype=np.float32)
        arm_high =  ARM_DELTA * np.ones(7, dtype=np.float32)
        grip_low  = np.array([0.0], dtype=np.float32)   # 0 = open
        grip_high = np.array([1.0], dtype=np.float32)   # 1 = close

        self.action_space = spaces.Box(
            low   = np.concatenate([arm_low,  grip_low]),
            high  = np.concatenate([arm_high, grip_high]),
            dtype = np.float32
        )

        # ── observation space ─────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (34,),
            dtype = np.float32
        )

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_pinch_pos(self):
        """3D position of gripper pinch site in world frame."""
        return self.data.site_xpos[self._pinch_site_id].copy()

    def _get_object_pos(self):
        """3D position of cube center in world frame."""
        return self.data.xpos[self._object_body_id].copy()

    def _get_target_pos(self):
        """3D position of target indicator in world frame (set in reset)."""
        return self.target_pos.copy()

    def _is_grasped(self):
        """
        Cube is considered grasped when:
        1. Pinch site is within GRASP_THRESHOLD of cube
        2. Gripper is closing (ctrl >= 100)
        """
        pinch_to_cube = np.linalg.norm(self._get_pinch_pos() - self._get_object_pos())
        gripper_closing = self.data.ctrl[7] >= 100.0

        return bool(pinch_to_cube < GRASP_THRESHOLD and gripper_closing)

    def _get_obs(self):
        """Build 35-dimensional observation vector."""
        arm_qpos    = self.data.qpos[:7].copy()        # 7 arm joints
        arm_qvel    = self.data.qvel[:7].copy()        # 7 arm velocities
        finger_qpos = self.data.qpos[7:9].copy()       # 2 finger joints

        pinch_pos   = self._get_pinch_pos()            # 3 gripper xyz
        object_pos  = self._get_object_pos()           # 3 cube xyz
        target_pos  = self._get_target_pos()           # 3 target xyz

        # error vectors — key signal for the agent
        pinch_to_obj  = object_pos - pinch_pos         # 3 — reach error
        obj_to_target = target_pos - object_pos        # 3 — place error

        gripper_state = np.array([self.gripper_state]) # 1 — open/close
        grasped_flag  = np.array([float(self._is_grasped())])  # 1
        cube_height   = np.array([object_pos[2] - TABLE_Z])    # 1 — lift progress

        return np.concatenate([
            arm_qpos,       # [0:7]
            arm_qvel,       # [7:14]
            finger_qpos,    # [14:16]
            pinch_pos,      # [16:19]
            object_pos,     # [19:22]
            target_pos,     # [22:25]
            pinch_to_obj,   # [25:28]
            obj_to_target,  # [28:31]
            gripper_state,  # [31]
            grasped_flag,   # [32]
            cube_height     # [33]  — note: shape is (35,) total but indices go to 34
        ]).astype(np.float32)

    def _compute_reward(self):
        pinch_pos  = self._get_pinch_pos()
        object_pos = self._get_object_pos()
        target_pos = self._get_target_pos()

        pinch_to_obj_dist  = float(np.linalg.norm(pinch_pos - object_pos))
        obj_to_target_dist = float(np.linalg.norm(object_pos - target_pos[:3]))
        cube_height        = object_pos[2] - TABLE_Z
        grasped            = self._is_grasped()

        reward = 0.0

        # stage 1 — reach: pull pinch toward cube always
        reward -= pinch_to_obj_dist * 2.0

        if pinch_to_obj_dist < 0.10:
            reward += 0.5
        if pinch_to_obj_dist < 0.05:
            reward += 1.0

        # velocity penalty — discourage violent motion
        joint_vel_penalty   = float(np.sum(np.abs(self.data.qvel[:7]))) * 0.01
        reward -= joint_vel_penalty

        # approach direction bonus — reward gripper being above cube
        if pinch_pos[2] > object_pos[2] + 0.02:
            reward += 0.3

        # stage 2 — grasp milestone (one-time bonus)
        if grasped:
            reward += 2.0
            if not self.has_grasped:
                reward += 50.0   # large one-time bonus first time grasped

        # stage 3 — lift milestone (one-time bonus)
        if grasped and cube_height > 0.01:
            reward += cube_height * 10.0
            if not self.has_lifted:
                reward += 50.0   # large one-time bonus first time lifted

        # stage 4 — transport (only if lifted)
        if self.has_lifted:
            reward -= obj_to_target_dist * 5.0
            if obj_to_target_dist < 0.10:
                reward += 2.0

        # stage 5 — place success
        if self._check_success():
            reward += 100.0

        # penalty if cube falls off table
        if object_pos[2] < TABLE_Z - 0.05:
            reward -= 20.0

        return reward

    def _check_success(self):
        object_pos  = self._get_object_pos()
        xy_dist     = float(np.linalg.norm(object_pos[:2] - self.target_pos[:2]))
        cube_height = object_pos[2] - TABLE_Z

        # cube must be near target AND must have been genuinely lifted
        # during this episode — eliminates false success from accidental pushes
        return bool(
            xy_dist < PLACE_THRESHOLD and
            cube_height < 0.06 and
            self.max_cube_height > 0.08
        )

    def _move_to_home(self):
        """Teleport arm to home configuration instantly."""
        self.data.qpos[:9] = HOME_QPOS.copy()
        self.data.qvel[:9] = 0.0
        self.data.ctrl[:]  = HOME_CTRL.copy()
        mujoco.mj_forward(self.model, self.data)

    def _spawn_object(self, pos):
        """Place cube at given xyz position with zero velocity."""
        # qpos for freejoint: [x, y, z, qw, qx, qy, qz]
        obj_qpos_start = 9           # after 7 arm + 2 finger joints
        self.data.qpos[obj_qpos_start:obj_qpos_start+3] = pos
        self.data.qpos[obj_qpos_start+3]  = 1.0   # qw
        self.data.qpos[obj_qpos_start+4:obj_qpos_start+7] = 0.0  # qx qy qz
        self.data.qvel[obj_qpos_start:obj_qpos_start+6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _move_target_indicator(self, pos):
        """Move the green target marker to given xyz position."""
        body_id = self._target_body_id
        self.model.body_pos[body_id] = pos.copy()

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset physics
        mujoco.mj_resetData(self.model, self.data)

        # arm to home pose
        self._move_to_home()

        # spawn cube at random position in object zone
        object_pos = np.random.uniform(OBJECT_LOW, OBJECT_HIGH)
        self._spawn_object(object_pos)

        # spawn target at random position in target zone
        # ensure target is not too close to object
        while True:
            target_xy = np.random.uniform(TARGET_LOW[:2], TARGET_HIGH[:2])
            if np.linalg.norm(target_xy - object_pos[:2]) > 0.10:
                break

        self.target_pos = np.array([target_xy[0], target_xy[1], TABLE_Z + 0.002],
                                   dtype=np.float32)
        self._move_target_indicator(self.target_pos)

        # reset episode state
        self.current_step  = 0
        self.max_cube_height = 0.0   # reset for new episode
        self.has_grasped = False
        self.has_lifted  = False
        self.gripper_state = 0.0
        self.phase         = 0

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # ── parse action ──────────────────────────────────────────────────────
        arm_deltas      = action[:7].astype(np.float64)
        gripper_command = float(action[7])   # 0=open, 1=close

        # ── apply arm deltas ──────────────────────────────────────────────────
        current_arm = self.data.ctrl[:7].copy()
        new_arm     = current_arm + arm_deltas

        # clip to joint limits
        for i in range(7):
            low  = self.model.actuator_ctrlrange[i, 0]
            high = self.model.actuator_ctrlrange[i, 1]
            new_arm[i] = np.clip(new_arm[i], low, high)

        self.data.ctrl[:7] = new_arm

        # ── apply gripper command ─────────────────────────────────────────────
        # gripper_command > 0.5 = close, else = open
        if gripper_command > 0.5:
            self.data.ctrl[7]  = GRIPPER_CTRL   # close
            self.gripper_state = 1.0
        else:
            self.data.ctrl[7]  = 0.0            # open
            self.gripper_state = 0.0

        # ── step physics ──────────────────────────────────────────────────────
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        cube_height          = self._get_object_pos()[2] - TABLE_Z
        self.max_cube_height = max(self.max_cube_height, cube_height)

        # update milestone flags
        if self._is_grasped() and not self.has_grasped:
            self.has_grasped = True

        if self.max_cube_height > 0.06 and not self.has_lifted:
            self.has_lifted = True

        # end episode immediately if cube falls off table
        if self._get_object_pos()[2] < TABLE_Z - 0.05:
            truncated = True

        # ── compute outcome ───────────────────────────────────────────────────
        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._check_success()
        truncated  = self.current_step >= MAX_EPISODE_STEPS

        info = {
            "is_success"       : terminated,
            "pinch_to_obj_dist": float(np.linalg.norm(
                self._get_pinch_pos() - self._get_object_pos())),
            "obj_to_tgt_dist"  : float(np.linalg.norm(
                self._get_object_pos() - self.target_pos)),
            "cube_height"      : float(self._get_object_pos()[2] - TABLE_Z),
            "grasped"          : self._is_grasped(),
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