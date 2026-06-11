import mujoco
import mujoco.viewer
import numpy as np
import time

from robot.observer   import RobotObserver
from config.robot_config import (
    TIMESTEP,
    DEFAULT_MOVE_STEPS,
    HOME_POSE,
    END_EFFECTOR_BODY
)


class RobotController:

    def __init__(self, model_path):
        self.model    = mujoco.MjModel.from_xml_path(model_path)
        self.data     = mujoco.MjData(self.model)
        self.observer = RobotObserver(self.model, self.data)
        self.viewer   = None

        print(f"Model loaded: {self.model.nq} joints")
        self._print_robot_info()

    def _print_robot_info(self):
        # prefixed with _ to signal this is internal/private
        print("\n--- Robot Info ---")
        print(f"Joints: {self.model.nq}  |  Actuators: {self.model.nu}")

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            low, high = self.model.jnt_range[i]
            print(f"  Joint {i}: {name:20s}  [{low:.2f}, {high:.2f}] rad")

        print("------------------\n")

    def is_safe(self, joint_index, angle):
        low, high = self.model.jnt_range[joint_index]

        return low <= angle <= high

    def set_joint_position(self, joint_index, angle):
        if not self.is_safe(joint_index, angle):
            low, high = self.model.jnt_range[joint_index]
            print(f"WARNING: Joint {joint_index} → {angle:.2f} rad "
                  f"outside [{low:.2f}, {high:.2f}]")

            return

        self.data.ctrl[joint_index] = angle

    def move_to_pose(self, target_angles, steps=DEFAULT_MOVE_STEPS):
        for i, angle in enumerate(target_angles):
            if not self.is_safe(i, angle):
                print(f"Pose rejected: joint {i} = {angle:.2f} rad "
                      f"outside safe range")

                return

        start_angles = self.observer.get_joint_positions().copy()

        for step in range(steps):
            t            = step / steps
            interpolated = start_angles + t * (
                np.array(target_angles) - start_angles
            )

            for i in range(self.model.nu):
                self.data.ctrl[i] = interpolated[i]

            self.step()
            self.viewer.sync()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            print("Simulation running. Close window to stop.\n")

            self.move_to_pose(HOME_POSE)

            while viewer.is_running():
                self.step()

                if int(self.data.time * 500) % 200 == 0:
                    obs = self.observer.get_observation(END_EFFECTOR_BODY)
                    print(f"t={self.data.time:.2f}s | "
                          f"EE={np.round(obs['end_effector_pos'], 3)}")

                viewer.sync()
                time.sleep(TIMESTEP)