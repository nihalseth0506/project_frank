import mujoco
import numpy as np


class RobotObserver:
    # handles all state reading from the simulation
    # separated from controller so observation logic
    # can be extended independently later (e.g. adding camera)

    def __init__(self, model, data):
        self.model = model
        self.data  = data

    def get_joint_positions(self):
        return self.data.qpos.copy()

    def get_joint_velocities(self):
        return self.data.qvel.copy()

    def get_end_effector_pos(self, body_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )

        return self.data.xpos[body_id].copy()

    def get_observation(self, end_effector_body):
        return {
            "joint_positions"  : self.get_joint_positions(),
            "joint_velocities" : self.get_joint_velocities(),
            "end_effector_pos" : self.get_end_effector_pos(end_effector_body)
        }