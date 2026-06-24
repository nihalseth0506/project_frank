"""
stages.py

Pick and place stage loop for FRANK Phase 3 VLA.
Contains the full 6-stage pipeline:
  0 APPROACH  — move above cube
  1 DESCEND   — lower fingers to cube
  2 GRASP     — close gripper
  3 LIFT      — raise cube
  4 TRANSPORT — move to tray
  5 RELEASE   — open gripper, check success
"""

import numpy as np
from scripts.vla.modules.vision import update_camera_window


TABLE_Z      = 0.530
CUBE_HALF_H  = 0.025
CUBE_Z       = TABLE_Z + CUBE_HALF_H

ABOVE_HEIGHT  = 0.08
LIFT_HEIGHT   = 0.08
PLACE_HEIGHT  = 0.08
FINGER_OFFSET = 0.117

SUCCESS_XY   = 0.09
SUCCESS_H    = 0.05


def run_pick_place_loop(env, max_steps=600):
    """
    Execute all pick and place stages.
    env must expose: object_pos, target_pos, stage, stage_step,
    _target_object_id, _target_colour, data, model,
    _reach_step(), _open_gripper(), _close_gripper(), _step(),
    _get_pinch(), _get_object_pos(), renderer, _cam_name.
    Returns True on success, False on timeout or failure.
    """
    for _ in range(max_steps * 8):

        # ── STAGE 0: approach above cube ──────────────────────────────
        if env.stage == 0:
            tgt    = env.object_pos.copy()
            tgt[2] = CUBE_Z + ABOVE_HEIGHT

            env._reach_step(tgt)
            env._open_gripper()

            pinch   = env._get_pinch()
            dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
            dist_z  = abs(pinch[2] - tgt[2])

            if dist_xy < 0.07 and dist_z < 0.06:
                print(f"  ✓ APPROACH — pinch {pinch.round(3)}")
                env.stage = 1; env.stage_step = 0
            elif env.stage_step >= max_steps:
                print(f"  ✗ APPROACH timeout (xy={dist_xy:.3f} z={dist_z:.3f})")
                return False

        # ── STAGE 1: descend to cube ──────────────────────────────────
        elif env.stage == 1:
            pinch = env._get_pinch()

            if env.stage_step % 10 == 0:
                update_camera_window(env._renderer, env.data, env._cam_name,
                                     f"STAGE 1: DESCEND  step={env.stage_step}",
                                     env._target_colour)

            pinch_target_z = env.object_pos[2] + 0.10 - FINGER_OFFSET
            tgt    = env.object_pos.copy()
            tgt[2] = max(pinch_target_z, pinch[2] - 0.08)

            env._reach_step(tgt, constrain_orientation=False)
            env._open_gripper()

            pinch    = env._get_pinch()
            finger_z = pinch[2] + FINGER_OFFSET
            dist_xy  = np.linalg.norm(pinch[:2] - env.object_pos[:2])

            print(f"    descend: finger z={finger_z:.3f} cube z={env.object_pos[2]:.3f} xy={dist_xy:.3f}")

            if finger_z < env.object_pos[2] + 0.05 and dist_xy < 0.04:
                print(f"  ✓ DESCEND — finger z={finger_z:.3f}")
                env.stage = 2; env.stage_step = 0
            elif env.stage_step >= max_steps:
                print(f"  ✗ DESCEND timeout")
                return False

        # ── STAGE 2: close gripper ────────────────────────────────────
        elif env.stage == 2:
            env.data.ctrl[:7] = env.data.qpos[:7].copy()
            env._close_gripper()
            env._step()

            fj1 = env.data.qpos[7]

            if env.stage_step >= 150:
                cube_h  = env._get_object_pos()[2] - TABLE_Z
                ee_dist = np.linalg.norm(env._get_pinch() - env._get_object_pos())
                print(f"  After close: cube_h={cube_h:.4f} ee={ee_dist:.3f} fj1={fj1:.4f}")
                env.stage = 3; env.stage_step = 0

        # ── STAGE 3: lift ─────────────────────────────────────────────
        elif env.stage == 3:
            if env.stage_step % 10 == 0:
                update_camera_window(env._renderer, env.data, env._cam_name,
                                     f"STAGE 3: LIFT  step={env.stage_step}",
                                     env._target_colour)

            cube_pos = env.data.xpos[env._target_object_id].copy()
            tgt      = np.array([cube_pos[0], cube_pos[1],
                                  TABLE_Z + LIFT_HEIGHT + 0.05])

            env._reach_step(tgt, constrain_orientation=False)
            env.data.ctrl[7] = 0.0

            cube_h = env._get_object_pos()[2] - TABLE_Z
            print(f"    lift: cube_h={cube_h:.3f}")

            if cube_h > LIFT_HEIGHT - 0.04:
                print(f"  ✓ LIFT — height {cube_h:.3f}m")
                env.stage = 4; env.stage_step = 0
            elif env.stage_step >= max_steps:
                print(f"  ✗ LIFT timeout — height {cube_h:.3f}m")
                return False

        # ── STAGE 4: transport ────────────────────────────────────────
        elif env.stage == 4:
            if env.stage_step % 10 == 0:
                update_camera_window(env._renderer, env.data, env._cam_name,
                                     f"STAGE 4: TRANSPORT  step={env.stage_step}",
                                     env._target_colour)

            tgt    = env.target_pos.copy()
            tgt[2] = TABLE_Z + PLACE_HEIGHT + 0.02

            env._reach_step(tgt, constrain_orientation=False)
            env.data.ctrl[7] = 0.0

            pinch   = env._get_pinch()
            dist_xy = np.linalg.norm(pinch[:2] - tgt[:2])
            cube_h  = env._get_object_pos()[2] - TABLE_Z

            print(f"    transport: pinch={pinch.round(3)} xy={dist_xy:.3f} cube_h={cube_h:.3f}")

            if dist_xy < 0.05:
                print(f"  ✓ TRANSPORT — releasing")
                env.stage = 5; env.stage_step = 0
            elif env.stage_step >= max_steps:
                if dist_xy < 0.10:
                    print(f"  ⚠ TRANSPORT partial (xy={dist_xy:.3f})")
                    env.stage = 5; env.stage_step = 0
                else:
                    print(f"  ✗ TRANSPORT timeout (xy={dist_xy:.3f})")
                    return False

        # ── STAGE 5: release ──────────────────────────────────────────
        elif env.stage == 5:
            env.data.ctrl[7] = min(env.data.ctrl[7] + 10.0, 255.0)
            env._step()

            if env.stage_step >= 50:
                cube_pos = env._get_object_pos()
                xy_dist  = np.linalg.norm(cube_pos[:2] - env.target_pos[:2])
                cube_h   = cube_pos[2] - TABLE_Z
                print(f"  RELEASE — cube→target={xy_dist:.4f}m  h={cube_h:.4f}m")
                return bool(xy_dist < SUCCESS_XY and cube_h < SUCCESS_H)

        env.stage_step += 1

    return False