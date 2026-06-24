"""
pick_place_scripted_env.py

Scripted pick and place — Panda reach policy + scripted gripper.
Phase 3: Multi-colour cubes, HSV colour classification, language-directed picking.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU for inference

import cv2
import numpy as np
import mujoco
import mujoco.viewer
import time
import math

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from ultralytics import YOLO

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

SCENE_PATH = os.path.join(_PROJ_ROOT, "models", "pick_place_scene.xml")


def _find_latest_panda_reach():
    panda_dir = os.path.join(_PROJ_ROOT, "models", "reach", "panda")
    best      = os.path.join(panda_dir, "trained_v5", "best")
    return os.path.join(best, "best_model"), os.path.join(best, "vec_normalize.pkl")


TABLE_Z     = 0.530
CUBE_HALF_H = 0.025
CUBE_Z      = TABLE_Z + CUBE_HALF_H   # 0.555

ABOVE_HEIGHT = 0.08
LIFT_HEIGHT  = 0.08
PLACE_HEIGHT = 0.08

HOME_QPOS = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.0])

TARGET_LOW  = np.array([0.45, -0.25, CUBE_Z])
TARGET_HIGH = np.array([0.55, -0.15, CUBE_Z])

COLOURS       = ['red', 'blue', 'yellow']
RETRACTED_QPOS = np.array([-1.5, -1.2, 0.0, -2.5, 0.0, 2.0, -0.7853])


class ScriptedPickPlace:

    def __init__(self, render=True, target_colour=None):
        """
        Args:
            render:         whether to open MuJoCo viewer
            target_colour:  fix target colour ('red'/'blue'/'yellow') or
                            None for random each episode
        """
        self.render_mode     = render
        self._fixed_colour   = target_colour

        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        self._pinch_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")
        self._tray_id   = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")

        # body IDs for all three cubes
        self._object_ids = {
            'red':    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object"),
            'blue':   mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_blue"),
            'yellow': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_yellow"),
        }

        # freejoint qpos start indices for each cube
        # arm: 7 joints + 2 fingers = 9, then each cube freejoint = 7 values
        self._qpos_starts = {
            'red':    9,
            'blue':   9 + 7,
            'yellow': 9 + 14,
        }

        model_path, norm_path = _find_latest_panda_reach()
        if model_path is None or not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                "No Panda reach policy found.\n"
                "Run: python scripts/train_reach_panda.py"
            )

        print(f"Loading policy : {model_path}.zip")
        self._reach_env, self._reach_model = self._load_policy(
            model_path, norm_path)
        print("Policy loaded.\n")

        self.stage             = 0
        self.stage_step        = 0
        self.target_pos        = np.zeros(3)
        self.object_pos        = np.zeros(3)
        self._target_colour    = None
        self._target_object_id = None

        YOLO_PATH = os.path.join(_PROJ_ROOT, "models", "yolo",
                                 "cube_detector_vla", "weights", "best.pt")

        # camera parameters — must match XML
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
        self._detector = YOLO(YOLO_PATH)
        print("YOLO detector loaded.")

    # ------------------------------------------------------------------ #
    #  Policy loading                                                      #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Motion helpers                                                      #
    # ------------------------------------------------------------------ #

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
            new_arm[i] = np.clip(
                new_arm[i],
                self.model.actuator_ctrlrange[i, 0],
                self.model.actuator_ctrlrange[i, 1]
            )

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

    # ------------------------------------------------------------------ #
    #  Spawning                                                            #
    # ------------------------------------------------------------------ #

    def _spawn_no_overlap(self):
        """Grid-based spawn — guaranteed 12cm separation."""
        # three fixed zones, randomise position within each zone
        zones = [
            ([0.50, 0.52], [-0.02, 0.01]),   # left zone
            ([0.58, 0.61], [ 0.01, 0.03]),   # right zone
            ([0.52, 0.56], [ 0.07, 0.09]),   # back zone
        ]
        np.random.shuffle(zones)  # randomise which colour goes to which zone

        positions = []
        for x_range, y_range in zones:
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            positions.append(np.array([x, y]))

        return positions

    # ------------------------------------------------------------------ #
    #  Vision                                                              #
    # ------------------------------------------------------------------ #

    def _classify_colour_hsv(self, crop_rgb):
        """Classify cube colour from RGB crop using HSV hue."""
        if crop_rgb.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        sat = np.median(hsv[:, :, 1])

        if sat < 50:
            return 'unknown'

        hue = np.median(hsv[:, :, 0])

        if hue < 10 or hue > 160:
            return 'red'
        if 20 < hue < 38:
            return 'yellow'
        if 95 < hue < 135:
            return 'blue'

        return 'unknown'

    def _detect_cube_pos(self):
        """
        Render overhead frame, run YOLO, classify each detection by HSV colour,
        return world position of the target colour cube.
        Falls back to ground truth if target colour not detected.
        """
        self._renderer.update_scene(self.data, camera=self._cam_name)
        frame   = self._renderer.render()
        results = self._detector(frame, verbose=False, conf=0.1)[0]

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame_bgr, f"Target: {self._target_colour}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        best_box  = None
        best_conf = 0.0

        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            crop   = frame[y1:y2, x1:x2]
            colour = self._classify_colour_hsv(crop)
            conf   = float(box.conf[0])
            print(f"    YOLO box: classified={colour} conf={conf:.3f}")

            is_target = (colour == self._target_colour)
            colour_bgr = (0, 255, 0) if is_target else (120, 120, 120)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)
            cv2.putText(frame_bgr, f"{colour} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1)

            if is_target and conf > best_conf:
                best_box  = box
                best_conf = conf

        cv2.imshow("FRANK — Overhead Camera", frame_bgr)
        cv2.waitKey(1)

        if best_box is None:
            print(f"  [{self._target_colour}] not detected — GT fallback")
            return self.data.xpos[self._target_object_id].copy()

        u, v = float(best_box.xywh[0][0]), float(best_box.xywh[0][1])
        wx, wy = self._pixel_to_world(u, v)

        return np.array([wx, wy, CUBE_Z])

    def _update_camera_window(self, label):
        """Refresh overhead camera window with stage label during execution."""
        self._renderer.update_scene(self.data, camera=self._cam_name)
        frame     = self._renderer.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame_bgr, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame_bgr, f"Target: {self._target_colour}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("FRANK — Overhead Camera", frame_bgr)
        cv2.waitKey(1)

    def _pixel_to_world(self, u, v):
        fovy_rad     = math.radians(self._fovy_deg)
        half_fov     = fovy_rad / 2.0
        aspect       = self._img_w / self._img_h
        height_above = self._cam_z - TABLE_Z
        scale_y      = math.tan(half_fov) * height_above
        scale_x      = scale_y * aspect

        u_norm =  (u / self._img_w) * 2.0 - 1.0
        v_norm =  (v / self._img_h) * 2.0 - 1.0

        wx = self._cam_x + u_norm * scale_x
        wy = self._cam_y - v_norm * scale_y

        return wx, wy

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS.copy()
        self.data.qvel[:9] = 0.0
        self.data.ctrl[:]  = HOME_CTRL.copy()
        mujoco.mj_forward(self.model, self.data)

        # choose target colour
        if self._fixed_colour is not None:
            self._target_colour = self._fixed_colour
        else:
            self._target_colour = np.random.choice(COLOURS)

        self._target_object_id = self._object_ids[self._target_colour]

        # spawn all three cubes with no overlap
        positions = self._spawn_no_overlap()

        for colour, pos in zip(COLOURS, positions):
            qs    = self._qpos_starts[colour]
            theta = np.random.uniform(0, 2 * np.pi)
            qw    = np.cos(theta / 2)
            qz    = np.sin(theta / 2)

            self.data.qpos[qs:qs+3] = [pos[0], pos[1], CUBE_Z]
            self.data.qpos[qs+3]    = 1.0
            self.data.qpos[qs+4]    = 0.0
            self.data.qpos[qs+5]    = 0.0
            self.data.qpos[qs+6]    = 0.0
            self.data.qvel[qs:qs+6] = 0.0

        # pick target xy for tray — away from all cubes
        target_cube_pos = positions[COLOURS.index(self._target_colour)]
        # for _ in range(100):
        #     tgt_xy = np.random.uniform(TARGET_LOW[:2], TARGET_HIGH[:2])
        #     if np.linalg.norm(tgt_xy - target_cube_pos) > 0.12:
        #         break

        # self.target_pos = np.array(
        #     [tgt_xy[0] + 0.03, tgt_xy[1], TABLE_Z + 0.002], dtype=np.float32)
        # self.model.body_pos[self._target_id] = self.target_pos.copy()

        # self.model.body_pos[self._tray_id] = np.array([
        #     tgt_xy[0], tgt_xy[1], TABLE_Z + 0.001
        # ])

        # fix tray at a consistent location for the session
        self.target_pos = np.array([0.55, -0.22, TABLE_Z + 0.002], dtype=np.float32)
        self.model.body_pos[self._target_id] = self.target_pos.copy()
        self.model.body_pos[self._tray_id]   = np.array([0.52, -0.22, TABLE_Z + 0.001])

        mujoco.mj_forward(self.model, self.data)

        # retract arm for clean overhead view
        saved_qpos = self.data.qpos[:9].copy()
        saved_ctrl = self.data.ctrl[:8].copy()

        self.data.qpos[:7] = RETRACTED_QPOS
        self.data.qpos[7]  = 0.04
        self.data.qpos[8]  = 0.04
        self.data.ctrl[:7] = RETRACTED_QPOS
        mujoco.mj_forward(self.model, self.data)

        # detect target cube from overhead camera
        detected = self._detect_cube_pos()

        # restore arm
        self.data.qpos[:9] = saved_qpos
        self.data.ctrl[:8] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)

        self.object_pos = detected
        self.stage      = 0
        self.stage_step = 0

        print(f"  Task   : pick the {self._target_colour} cube")
        print(f"  Cube   : {self.object_pos.round(3)}")
        print(f"  Target : {self.target_pos.round(3)}")
        print(f"  Pinch  : {self._get_pinch().round(3)}")

    # ------------------------------------------------------------------ #
    #  Episode                                                             #
    # ------------------------------------------------------------------ #

    def run_episode(self, max_steps=600):
        self.reset()

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # warmup — settle arm at home
        for _ in range(100):
            self.data.qpos[:7] = HOME_QPOS[:7].copy()
            self.data.ctrl[:7] = HOME_CTRL[:7].copy()
            self.data.ctrl[7]  = 255.0
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            if self.render_mode and self.viewer is not None:
                self.viewer.sync()
                time.sleep(0.005)

        for _ in range(max_steps * 8):

            # STAGE 0: approach above cube
            if self.stage == 0:
                tgt    = self.object_pos.copy()
                tgt[2] = CUBE_Z + ABOVE_HEIGHT

                self._reach_step(tgt)
                self._open_gripper()

                pinch   = self._get_pinch()
                dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
                dist_z  = abs(pinch[2] - tgt[2])

                if dist_xy < 0.07 and dist_z < 0.06:
                    print(f"  ✓ APPROACH — pinch {pinch.round(3)}")
                    self.stage = 1; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ APPROACH timeout (xy={dist_xy:.3f} z={dist_z:.3f})")
                    break

            # STAGE 1: descend to cube
            elif self.stage == 1:
                pinch = self._get_pinch()    # ← this line must be first

                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 1: DESCEND  step={self.stage_step}")

                FINGER_OFFSET  = 0.117
                pinch_target_z = self.object_pos[2] + 0.10 - FINGER_OFFSET

                tgt    = self.object_pos.copy()
                tgt[2] = max(pinch_target_z, pinch[2] - 0.08)

                self._reach_step(tgt, constrain_orientation=False)
                self._open_gripper()

                pinch    = self._get_pinch()
                finger_z = pinch[2] + FINGER_OFFSET
                dist_xy  = np.linalg.norm(pinch[:2] - self.object_pos[:2])

                print(f"    descend: finger z={finger_z:.3f} cube z={self.object_pos[2]:.3f} xy={dist_xy:.3f}")

                if finger_z < self.object_pos[2] + 0.05 and dist_xy < 0.04:
                    print(f"  ✓ DESCEND — finger z={finger_z:.3f}")
                    self.stage = 2; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ DESCEND timeout")
                    break

            # STAGE 2: close gripper
            elif self.stage == 2:
                self.data.ctrl[:7] = self.data.qpos[:7].copy()
                self._close_gripper()
                self._step()

                fj1 = self.data.qpos[7]

                if self.stage_step >= 150:
                    cube_h  = self._get_object_pos()[2] - TABLE_Z
                    ee_dist = np.linalg.norm(self._get_pinch() - self._get_object_pos())
                    print(f"  After close: cube_h={cube_h:.4f} ee={ee_dist:.3f} fj1={fj1:.4f} steps={self.stage_step}")
                    self.stage = 3; self.stage_step = 0

            # STAGE 3: lift
            elif self.stage == 3:
                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 3: LIFT  step={self.stage_step}")

                cube_pos = self.data.xpos[self._target_object_id].copy()
                tgt      = np.array([cube_pos[0], cube_pos[1], TABLE_Z + LIFT_HEIGHT + 0.05])

                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0

                cube_h = self._get_object_pos()[2] - TABLE_Z
                print(f"    lift: cube_h={cube_h:.3f}")

                if cube_h > LIFT_HEIGHT - 0.04:
                    print(f"  ✓ LIFT — height {cube_h:.3f}m")
                    self.stage = 4; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ LIFT timeout — height {cube_h:.3f}m")
                    break

            # STAGE 4: transport
            elif self.stage == 4:
                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 4: TRANSPORT  step={self.stage_step}")

                tgt    = self.target_pos.copy()
                tgt[2] = TABLE_Z + PLACE_HEIGHT + 0.02
                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0

                pinch   = self._get_pinch()
                dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
                cube_h  = self._get_object_pos()[2] - TABLE_Z

                print(f"    transport: pinch={pinch.round(3)} xy={dist_xy:.3f} cube_h={cube_h:.3f}")

                if dist_xy < 0.05:
                    print(f"  ✓ TRANSPORT — releasing")
                    self.stage = 5; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    if dist_xy < 0.10:
                        print(f"  ⚠ TRANSPORT partial — releasing anyway (xy={dist_xy:.3f})")
                        self.stage = 5; self.stage_step = 0
                    else:
                        print(f"  ✗ TRANSPORT timeout (xy={dist_xy:.3f})")
                        break

            # STAGE 5: release
            elif self.stage == 5:
                current_ctrl      = self.data.ctrl[7]
                self.data.ctrl[7] = min(current_ctrl + 10.0, 255.0)
                self._step()

                if self.stage_step >= 50:
                    cube_pos = self._get_object_pos()
                    xy_dist  = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
                    cube_h   = cube_pos[2] - TABLE_Z
                    print(f"  RELEASE — cube→target={xy_dist:.4f}m  h={cube_h:.4f}m")
                    success = bool(xy_dist < 0.09 and cube_h < 0.05)
                    return success

            self.stage_step += 1

        return False

    def _return_to_home(self):
        """Return arm to home pose."""
        for _ in range(100):
            self.data.qpos[:7] = HOME_QPOS[:7].copy()
            self.data.ctrl[:7] = HOME_CTRL[:7].copy()
            self.data.ctrl[7]  = 255.0
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            if self.render_mode and self.viewer is not None:
                self.viewer.sync()
                time.sleep(0.005)

    def pick_colour(self, colour, max_steps=600):
        self._target_colour    = colour
        self._target_object_id = self._object_ids[colour]
        self._fixed_colour     = colour

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # retract arm, detect cube, restore arm
        saved_qpos = self.data.qpos[:9].copy()
        saved_ctrl = self.data.ctrl[:8].copy()

        self.data.qpos[:7] = RETRACTED_QPOS
        self.data.qpos[7]  = 0.04
        self.data.qpos[8]  = 0.04
        self.data.ctrl[:7] = RETRACTED_QPOS
        mujoco.mj_forward(self.model, self.data)

        self.object_pos = self._detect_cube_pos()

        self.data.qpos[:9] = saved_qpos
        self.data.ctrl[:8] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)

        # tray position stays fixed from reset() — do not move it here

        self.stage      = 0
        self.stage_step = 0

        print(f"  Cube   : {self.object_pos.round(3)}")
        print(f"  Target : {self.target_pos.round(3)}")

        self._return_to_home()
        success = self._run_pick_place_loop(max_steps)
        self._return_to_home()

        return success

    def _run_pick_place_loop(self, max_steps=600):
        """Execute the pick and place stage loop. Returns True on success."""
        for _ in range(max_steps * 8):

            if self.stage == 0:
                tgt    = self.object_pos.copy()
                tgt[2] = CUBE_Z + ABOVE_HEIGHT
                self._reach_step(tgt)
                self._open_gripper()
                pinch   = self._get_pinch()
                dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
                dist_z  = abs(pinch[2] - tgt[2])
                if dist_xy < 0.07 and dist_z < 0.06:
                    print(f"  ✓ APPROACH — pinch {pinch.round(3)}")
                    self.stage = 1; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ APPROACH timeout")
                    return False

            elif self.stage == 1:
                pinch = self._get_pinch()
                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 1: DESCEND  step={self.stage_step}")
                FINGER_OFFSET  = 0.117
                pinch_target_z = self.object_pos[2] + 0.10 - FINGER_OFFSET
                tgt    = self.object_pos.copy()
                tgt[2] = max(pinch_target_z, pinch[2] - 0.08)
                self._reach_step(tgt, constrain_orientation=False)
                self._open_gripper()
                pinch    = self._get_pinch()
                finger_z = pinch[2] + FINGER_OFFSET
                dist_xy  = np.linalg.norm(pinch[:2] - self.object_pos[:2])
                print(f"    descend: finger z={finger_z:.3f} cube z={self.object_pos[2]:.3f} xy={dist_xy:.3f}")
                if finger_z < self.object_pos[2] + 0.05 and dist_xy < 0.04:
                    print(f"  ✓ DESCEND — finger z={finger_z:.3f}")
                    self.stage = 2; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ DESCEND timeout")
                    return False

            elif self.stage == 2:
                self.data.ctrl[:7] = self.data.qpos[:7].copy()
                self._close_gripper()
                self._step()
                fj1 = self.data.qpos[7]
                if self.stage_step >= 150:
                    cube_h  = self._get_object_pos()[2] - TABLE_Z
                    ee_dist = np.linalg.norm(self._get_pinch() - self._get_object_pos())
                    print(f"  After close: cube_h={cube_h:.4f} ee={ee_dist:.3f} fj1={fj1:.4f}")
                    self.stage = 3; self.stage_step = 0

            elif self.stage == 3:
                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 3: LIFT  step={self.stage_step}")
                cube_pos = self.data.xpos[self._target_object_id].copy()
                tgt      = np.array([cube_pos[0], cube_pos[1], TABLE_Z + LIFT_HEIGHT + 0.05])
                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0
                cube_h = self._get_object_pos()[2] - TABLE_Z
                print(f"    lift: cube_h={cube_h:.3f}")
                if cube_h > LIFT_HEIGHT - 0.04:
                    print(f"  ✓ LIFT — height {cube_h:.3f}m")
                    self.stage = 4; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    print(f"  ✗ LIFT timeout")
                    return False

            elif self.stage == 4:
                if self.stage_step % 10 == 0:
                    self._update_camera_window(f"STAGE 4: TRANSPORT  step={self.stage_step}")
                tgt    = self.target_pos.copy()
                tgt[2] = TABLE_Z + PLACE_HEIGHT + 0.02
                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0
                pinch   = self._get_pinch()
                dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
                cube_h  = self._get_object_pos()[2] - TABLE_Z
                print(f"    transport: pinch={pinch.round(3)} xy={dist_xy:.3f} cube_h={cube_h:.3f}")
                if dist_xy < 0.05:
                    print(f"  ✓ TRANSPORT — releasing")
                    self.stage = 5; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    if dist_xy < 0.10:
                        print(f"  ⚠ TRANSPORT partial")
                        self.stage = 5; self.stage_step = 0
                    else:
                        print(f"  ✗ TRANSPORT timeout")
                        return False

            elif self.stage == 5:
                current_ctrl      = self.data.ctrl[7]
                self.data.ctrl[7] = min(current_ctrl + 10.0, 255.0)
                self._step()
                if self.stage_step >= 50:
                    cube_pos = self._get_object_pos()
                    xy_dist  = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
                    cube_h   = cube_pos[2] - TABLE_Z
                    print(f"  RELEASE — cube→target={xy_dist:.4f}m  h={cube_h:.4f}m")
                    return bool(xy_dist < 0.09 and cube_h < 0.05)

            self.stage_step += 1

        return False

    def close(self):
        cv2.destroyAllWindows()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None