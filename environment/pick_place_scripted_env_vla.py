"""
pick_place_scripted_env_vla.py

FRANK Phase 3 — VLA Pick and Place
Multi-colour cubes, YOLO detection, HSV colour classification,
language-directed picking via PPO reach policy.

Architecture:
    Vision   — scripts/vla/modules/vision.py
    Spawning — scripts/vla/modules/spawner.py
    Stages   — scripts/vla/modules/stages.py
    Motion   — _reach_step, _return_to_home (inline, policy-coupled)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from ultralytics import YOLO

from scripts.vla.modules.vision   import detect_cube_pos
from scripts.vla.modules.spawner  import spawn_no_overlap
from scripts.vla.modules.stages   import run_pick_place_loop

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")

# ── constants ─────────────────────────────────────────────────────────────────

TABLE_Z     = 0.530
CUBE_HALF_H = 0.025
CUBE_Z      = TABLE_Z + CUBE_HALF_H   # 0.555

ABOVE_HEIGHT = 0.08
LIFT_HEIGHT  = 0.08
PLACE_HEIGHT = 0.08

HOME_QPOS = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.0])

# fixed tray position for the full session
TRAY_POS        = np.array([0.52, -0.22, TABLE_Z + 0.001])
TARGET_POS      = np.array([0.55, -0.22, TABLE_Z + 0.002])

# random tray bounds (kept for optional use)
TARGET_LOW  = np.array([0.45, -0.25, CUBE_Z])
TARGET_HIGH = np.array([0.55, -0.15, CUBE_Z])

COLOURS        = ['red', 'blue', 'yellow']
RETRACTED_QPOS = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853])

YOLO_PATH = os.path.join(_PROJ_ROOT, "models", "yolo",
                          "cube_detector_vla", "weights", "best.pt")


def _find_policy():
    best = os.path.join(_PROJ_ROOT, "models", "reach", "panda", "trained_v5", "best")
    return os.path.join(best, "best_model"), os.path.join(best, "vec_normalize.pkl")


# ── environment ───────────────────────────────────────────────────────────────

class ScriptedPickPlace:

    def __init__(self, render=True, target_colour=None, random_tray=False):
        """
        Args:
            render:        open MuJoCo viewer
            target_colour: fix colour per episode ('red'/'blue'/'yellow') or None for random
            random_tray:   if True randomise tray position each pick, else fixed
        """
        self.render_mode   = render
        self._fixed_colour = target_colour
        self._random_tray  = random_tray

        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        # site and body IDs
        self._pinch_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")
        self._tray_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")

        self._object_ids = {
            'red':    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object"),
            'blue':   mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_blue"),
            'yellow': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_yellow"),
        }

        self._qpos_starts = {'red': 9, 'blue': 16, 'yellow': 23}

        # state
        self.stage             = 0
        self.stage_step        = 0
        self.target_pos        = np.zeros(3)
        self.object_pos        = np.zeros(3)
        self._target_colour    = None
        self._target_object_id = None

        # camera
        self._cam_name = "overhead_cam"
        self._img_w    = 640
        self._img_h    = 480
        self._fovy_deg = 60.0
        self._cam_x    = 0.5
        self._cam_y    = -0.05
        self._cam_z    = 1.5

        self._renderer = mujoco.Renderer(self.model,
                                          height=self._img_h,
                                          width=self._img_w)

        # load models
        model_path, norm_path = _find_policy()
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Reach policy not found: {model_path}.zip")

        print(f"Loading policy : {model_path}.zip")
        self._reach_env, self._reach_model = self._load_policy(model_path, norm_path)
        print("Policy loaded.\n")

        self._detector = YOLO(YOLO_PATH)
        print("YOLO detector loaded.")

    # ── policy ────────────────────────────────────────────────────────────────

    def _load_policy(self, model_path, norm_path):
        from environment.reach_env_panda import PandaReachEnv

        def make_env():
            return Monitor(PandaReachEnv(render_mode=None))

        env             = make_vec_env(make_env, n_envs=1)
        env             = VecNormalize.load(norm_path, env)
        env.training    = False
        env.norm_reward = False
        model           = PPO.load(model_path, env=env, device="cpu")

        return env, model

    # ── motion ────────────────────────────────────────────────────────────────

    def _reach_step(self, target_xyz, constrain_orientation=True):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),
            self.data.qvel[:7].copy(),
            self.data.site_xpos[self._pinch_id].copy(),
            np.array(target_xyz, dtype=np.float32)
        ]).astype(np.float32)

        obs_norm  = self._reach_env.normalize_obs(obs.reshape(1, -1))
        action, _ = self._reach_model.predict(obs_norm, deterministic=True)
        deltas    = action[0][:7].astype(np.float64)

        current = self.data.ctrl[:7].copy()
        new_arm = current + deltas

        if constrain_orientation:
            new_arm[5] = np.clip(new_arm[5], 1.85, 1.95)
            new_arm[6] = np.clip(new_arm[6], -0.1, 0.1)

        for i in range(7):
            new_arm[i] = np.clip(new_arm[i],
                                  self.model.actuator_ctrlrange[i, 0],
                                  self.model.actuator_ctrlrange[i, 1])

        self.data.ctrl[:7] = new_arm

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()
            time.sleep(0.002)

    def _get_pinch(self):
        return self.data.site_xpos[self._pinch_id].copy()

    def _get_object_pos(self):
        return self.data.xpos[self._target_object_id].copy()

    def _open_gripper(self):
        self.data.ctrl[7] = 255.0

    def _close_gripper(self):
        self.data.ctrl[7] = 0.0

    def _step(self):
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()
            time.sleep(0.002)

    def _return_to_home(self):
        for _ in range(100):
            self.data.qpos[:7] = HOME_QPOS[:7].copy()
            self.data.ctrl[:7] = HOME_CTRL[:7].copy()
            self.data.ctrl[7]  = 255.0
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            if self.render_mode and self.viewer is not None:
                self.viewer.sync()
                time.sleep(0.005)

    def _detect(self):
        """Retract arm, detect target cube, restore arm."""
        saved_qpos = self.data.qpos[:9].copy()
        saved_ctrl = self.data.ctrl[:8].copy()

        self.data.qpos[:7] = RETRACTED_QPOS
        self.data.qpos[7]  = 0.04
        self.data.qpos[8]  = 0.04
        self.data.ctrl[:7] = RETRACTED_QPOS
        mujoco.mj_forward(self.model, self.data)

        pos = detect_cube_pos(
            self._renderer, self._detector, self.data,
            self._cam_name, self._img_w, self._img_h,
            self._fovy_deg, self._cam_x, self._cam_y, self._cam_z,
            TABLE_Z, CUBE_Z, self._target_colour, self._target_object_id
        )

        self.data.qpos[:9] = saved_qpos
        self.data.ctrl[:8] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)

        return pos

    def _set_tray(self, cube_positions=None):
        """
        Set tray position. Fixed by default; random if self._random_tray is True.
        cube_positions: list of (x,y) used to avoid overlap when random.
        """
        if self._random_tray and cube_positions is not None:
            target_cube_pos = cube_positions[COLOURS.index(self._target_colour)]
            for _ in range(100):
                tgt_xy = np.random.uniform(TARGET_LOW[:2], TARGET_HIGH[:2])
                if np.linalg.norm(tgt_xy - target_cube_pos) > 0.12:
                    break
            self.target_pos = np.array(
                [tgt_xy[0] + 0.03, tgt_xy[1], TABLE_Z + 0.002], dtype=np.float32)
            self.model.body_pos[self._tray_id] = np.array(
                [tgt_xy[0], tgt_xy[1], TABLE_Z + 0.001])
        else:
            self.target_pos = TARGET_POS.astype(np.float32)
            self.model.body_pos[self._tray_id] = TRAY_POS

        self.model.body_pos[self._target_id] = self.target_pos.copy()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self):
        """
        Full reset — spawns all three cubes, sets tray, detects target cube.
        Used at session start and between independent episodes.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS.copy()
        self.data.qvel[:9] = 0.0
        self.data.ctrl[:]  = HOME_CTRL.copy()
        mujoco.mj_forward(self.model, self.data)

        self._target_colour = (self._fixed_colour
                               if self._fixed_colour is not None
                               else np.random.choice(COLOURS))
        self._target_object_id = self._object_ids[self._target_colour]

        positions = spawn_no_overlap()
        for colour, pos in zip(COLOURS, positions):
            qs = self._qpos_starts[colour]
            self.data.qpos[qs:qs+3] = [pos[0], pos[1], CUBE_Z]
            self.data.qpos[qs+3]    = 1.0
            self.data.qpos[qs+4:qs+7] = 0.0
            self.data.qvel[qs:qs+6] = 0.0

        self._set_tray(positions)
        mujoco.mj_forward(self.model, self.data)

        self.object_pos = self._detect()
        self.stage      = 0
        self.stage_step = 0

        print(f"  Task   : pick the {self._target_colour} cube")
        print(f"  Cube   : {self.object_pos.round(3)}")
        print(f"  Target : {self.target_pos.round(3)}")
        print(f"  Pinch  : {self._get_pinch().round(3)}")

    def pick_colour(self, colour, max_steps=600):
        """
        Pick a specific colour cube without resetting the scene.
        Used by run_vla.py for sequential multi-cube picking.
        """
        self._target_colour    = colour
        self._target_object_id = self._object_ids[colour]
        self._fixed_colour     = colour

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.object_pos = self._detect()
        self.stage      = 0
        self.stage_step = 0

        print(f"  Cube   : {self.object_pos.round(3)}")
        print(f"  Target : {self.target_pos.round(3)}")

        self._return_to_home()
        success = run_pick_place_loop(self, max_steps)
        self._return_to_home()

        return success

    def run_episode(self, max_steps=600):
        """
        Full episode — reset scene then pick target cube.
        Used for automated testing and success rate evaluation.
        """
        self.reset()

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self._return_to_home()
        return run_pick_place_loop(self, max_steps)

    def close(self):
        cv2.destroyAllWindows()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None