"""
pick_place_scripted_env.py

Scripted pick and place — Panda reach policy + scripted gripper.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU for inference

import numpy as np
import mujoco
import mujoco.viewer
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

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

OBJECT_LOW  = np.array([0.50, -0.02, CUBE_Z])   # push x start further out
OBJECT_HIGH = np.array([0.60,  0.10, CUBE_Z])   # keep same x end

TARGET_LOW  = np.array([0.45, -0.25, CUBE_Z])
TARGET_HIGH = np.array([0.55, -0.15, CUBE_Z])


class ScriptedPickPlace:

    def __init__(self, render=True):
        self.render_mode = render

        self.model  = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data   = mujoco.MjData(self.model)
        self.viewer = None

        self._pinch_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_indicator")

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

        self.stage      = 0
        self.stage_step = 0
        self.target_pos = np.zeros(3)
        self.object_pos = np.zeros(3)

        self._tray_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")

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
        return self.data.xpos[self._object_id].copy()

    def _is_grasped(self):
        dist    = np.linalg.norm(self._get_pinch() - self._get_object_pos())
        closing = self.data.ctrl[7] <= 50.0   # ← 0 = closed, so check <= 50
        return bool(dist < 0.15 and closing)

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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = HOME_QPOS.copy()
        self.data.qvel[:9] = 0.0
        self.data.ctrl[:]  = HOME_CTRL.copy()
        mujoco.mj_forward(self.model, self.data)

        obj_pos = np.random.uniform(OBJECT_LOW, OBJECT_HIGH)
        qs = 9
        self.data.qpos[qs:qs+3]   = obj_pos
        self.data.qpos[qs+3]      = 1.0
        self.data.qpos[qs+4:qs+7] = 0.0
        self.data.qvel[qs:qs+6]   = 0.0

        for _ in range(100):
            tgt_xy = np.random.uniform(TARGET_LOW[:2], TARGET_HIGH[:2])
            if np.linalg.norm(tgt_xy - obj_pos[:2]) > 0.12:
                break

        self.target_pos = np.array(
            [tgt_xy[0] + 0.03, tgt_xy[1], TABLE_Z + 0.002], dtype=np.float32)
        self.model.body_pos[self._target_id] = self.target_pos.copy()
        mujoco.mj_forward(self.model, self.data)

        # random z rotation for cube
        theta = np.random.uniform(0, 2 * np.pi)
        
        qw    = np.cos(theta / 2)
        qz    = np.sin(theta / 2)

        self.data.qpos[qs:qs+3]   = obj_pos
        self.data.qpos[qs+3]      = qw   # w
        self.data.qpos[qs+4]      = 0.0  # x
        self.data.qpos[qs+5]      = 0.0  # y
        self.data.qpos[qs+6]      = qz   # z
        self.data.qvel[qs:qs+6]   = 0.0

        # in reset(), after setting target_pos:
        self._tray_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")

        # move tray to target position
        self.model.body_pos[self._tray_id] = np.array([
            tgt_xy[0], tgt_xy[1], TABLE_Z + 0.001
        ])

        self.object_pos = self._get_object_pos().copy()
        self.stage      = 0
        self.stage_step = 0

        print(f"  Cube   : {self.object_pos.round(3)}")
        print(f"  Target : {self.target_pos.round(3)}")
        print(f"  Pinch  : {self._get_pinch().round(3)}")

    def run_episode(self, max_steps=600):
        self.reset()

        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # warmup — settle arm at home before policy starts
        for _ in range(50):
            self.data.ctrl[:7] = HOME_CTRL[:7].copy()
            self.data.ctrl[7]  = 255.0
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            if self.render_mode and self.viewer is not None:
                self.viewer.sync()
                time.sleep(0.002)

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
                self.object_pos = self._get_object_pos().copy()
                pinch           = self._get_pinch()

                FINGER_OFFSET   = 0.117
                pinch_target_z  = self.object_pos[2] + 0.10 - FINGER_OFFSET

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

            # STAGE 2: close gripper — freeze arm, just close
            elif self.stage == 2:
                self.data.ctrl[:7] = self.data.qpos[:7].copy()  # hold position
                self._close_gripper()
                self._step()

                fj1 = self.data.qpos[7]

                # exit when fingers closed enough OR max wait reached
                if self.stage_step >= 275:
                    cube_h  = self._get_object_pos()[2] - TABLE_Z
                    ee_dist = np.linalg.norm(self._get_pinch() - self._get_object_pos())
                    print(f"  After close: cube_h={cube_h:.4f} ee={ee_dist:.3f} fj1={fj1:.4f} steps={self.stage_step}")
                    self.stage = 3; self.stage_step = 0

            # STAGE 3: lift — scripted joint2
            elif self.stage == 3:
                cube_pos = self._get_object_pos().copy()
                # target = current cube xy, but higher z
                tgt = np.array([cube_pos[0], cube_pos[1], TABLE_Z + LIFT_HEIGHT + 0.05])

                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0   # keep gripper closed directly

                cube_h = self._get_object_pos()[2] - TABLE_Z
                print(f"    lift: cube_h={cube_h:.3f}")

                if cube_h > LIFT_HEIGHT - 0.04:
                    print(f"  ✓ LIFT — height {cube_h:.3f}m")
                    self.stage = 4; self.stage_step = 0  # ← transition, don't break
                elif self.stage_step >= max_steps:
                    print(f"  ✗ LIFT timeout — height {cube_h:.3f}m")
                    break
            
            elif self.stage == 4:
                tgt    = self.target_pos.copy()
                tgt[2] = TABLE_Z + PLACE_HEIGHT + 0.02
                self._reach_step(tgt, constrain_orientation=False)
                self.data.ctrl[7] = 0.0

                pinch   = self._get_pinch()
                dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
                dist_z  = abs(pinch[2] - tgt[2])
                cube_h  = self._get_object_pos()[2] - TABLE_Z

                print(f"    transport: pinch={pinch.round(3)} xy={dist_xy:.3f} cube_h={cube_h:.3f}")

                # release when xy is close — don't wait for perfect z alignment
                if dist_xy < 0.05:
                    print(f"  ✓ TRANSPORT — releasing")
                    self.stage = 5; self.stage_step = 0
                elif self.stage_step >= max_steps:
                    # timeout but still release if reasonably close
                    if dist_xy < 0.10:
                        print(f"  ⚠ TRANSPORT partial — releasing anyway (xy={dist_xy:.3f})")
                        self.stage = 5; self.stage_step = 0
                    else:
                        print(f"  ✗ TRANSPORT timeout (xy={dist_xy:.3f})")
                        break

            elif self.stage == 5:
                # ramp gripper open slowly to avoid flinging cube
                current_ctrl = self.data.ctrl[7]
                self.data.ctrl[7] = min(current_ctrl + 10.0, 255.0)  # open 10 units per step
                self._step()

                if self.stage_step >= 50:
                    cube_pos = self._get_object_pos()
                    xy_dist  = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
                    cube_h   = cube_pos[2] - TABLE_Z
                    print(f"  RELEASE — cube→target={xy_dist:.4f}m  h={cube_h:.4f}m")
                    success  = bool(xy_dist < 0.09 and cube_h < 0.05)
                    return success

            self.stage_step += 1

        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None