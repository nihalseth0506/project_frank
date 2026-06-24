"""
run_vla.py

FRANK Phase 3 — VLA Pick and Place
Type colour instruction in the OpenCV window, press Enter to execute.
All three cubes picked in sequence within a single session.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import argparse

import mujoco
from environment.pick_place_scripted_env_vla import (
    ScriptedPickPlace, HOME_QPOS, HOME_CTRL, RETRACTED_QPOS
)

from scripts.vla.modules.vision import update_camera_window


WINDOW_NAME = "FRANK — Overhead Camera"


def parse_colour(instruction):
    instruction = instruction.strip().lower()
    if "red"    in instruction: return "red"
    if "blue"   in instruction: return "blue"
    if "yellow" in instruction: return "yellow"
    return None


def get_instruction_from_screen():
    """
    Show text input overlay on OpenCV window.
    User types colour and presses Enter.
    Returns parsed colour or None if Esc pressed.
    """
    typed  = ""
    colour = None

    while colour is None:
        display = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.putText(display, "FRANK  VLA Phase 3", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(display, "Type colour and press Enter:", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(display, "red   /   blue   /   yellow", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 150), 1)

        cv2.rectangle(display, (10, 210), (630, 265), (40, 40, 40), -1)
        cv2.rectangle(display, (10, 210), (630, 265), (200, 200, 200), 1)
        cv2.putText(display, f"> {typed}_", (22, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(display, "Esc to quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:             # Esc
            return None
        elif key == 13:           # Enter
            colour = parse_colour(typed)
            if colour is None:
                err = display.copy()
                cv2.putText(err, "Not understood — try again", (10, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(WINDOW_NAME, err)
                cv2.waitKey(1200)
                typed = ""
        elif key == 8:            # Backspace
            typed = typed[:-1]
        elif 32 <= key <= 126:    # printable
            typed += chr(key)

    return colour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    print("=" * 55)
    print("FRANK — VLA Pick and Place (Phase 3)")
    print("Vision + Language + Panda reach policy")
    print("=" * 55)

    controller = ScriptedPickPlace(render=not args.no_render)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # spawn scene
    controller.reset()

    # open viewer BEFORE asking for input so user can adjust camera angle
    if not args.no_render and controller.viewer is None:
        controller.viewer = mujoco.viewer.launch_passive(
            controller.model, controller.data)

    # warmup — hold home pose so viewer shows clean scene
    for _ in range(150):
        controller.data.qpos[:7] = controller.model.key_qpos[0][:7] \
            if controller.model.nkey > 0 else HOME_QPOS[:7]
        controller.data.ctrl[:7] = HOME_CTRL[:7]
        controller.data.ctrl[7]  = 255.0
        for _ in range(5):
            import mujoco as _mj
            _mj.mj_step(controller.model, controller.data)
        if controller.viewer is not None:
            controller.viewer.sync()

    # show detection frame so user sees the scene in camera window too
    controller._target_colour = 'red'   # dummy for display
    controller._target_object_id = controller._object_ids['red']
    saved = controller.data.qpos[:9].copy()
    ctrl_saved = controller.data.ctrl[:8].copy()
    controller.data.qpos[:7] = RETRACTED_QPOS
    import mujoco as _mj
    _mj.mj_forward(controller.model, controller.data)
    update_camera_window(
        controller._renderer, controller.data, controller._cam_name,
        "Adjust viewer — then type colour below",
        controller._target_colour or "none"
    )
    controller.data.qpos[:9] = saved
    controller.data.ctrl[:8] = ctrl_saved
    _mj.mj_forward(controller.model, controller.data)

    print("\nScene ready. Adjust the viewer angle, then type in the camera window.")
    print("Press any key in camera window when ready.\n")
    cv2.waitKey(0)   # wait for keypress before showing input prompt

    picked        = []
    total_success = 0

    while True:
        colour = get_instruction_from_screen()

        if colour is None:
            print("Quit.")
            break

        if colour in picked:
            msg = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(msg, f"{colour} already placed!", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
            cv2.imshow(WINDOW_NAME, msg)
            cv2.waitKey(1500)
            continue

        print(f"\n{'─' * 40}")
        print(f"Pick {len(picked)+1}/3 — {colour} cube")

        success = controller.pick_colour(colour, max_steps=args.max_steps)
        total_success += int(success)

        if success:
            picked.append(colour)
            print(f"Result: ✅ SUCCESS — {colour} placed ({len(picked)}/3)")
            if len(picked) == 3:
                done = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(done, "All 3 cubes placed!", (80, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(done, "Press Esc to quit", (160, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                cv2.imshow(WINDOW_NAME, done)
                cv2.waitKey(3000)
                break
        else:
            print(f"Result: ❌ FAILED — type same colour to retry")

    print(f"\n{'═' * 55}")
    print(f"Picked : {picked}")
    print(f"Success: {total_success}/{len(picked) if picked else 3}")

    controller.close()


if __name__ == "__main__":
    main()