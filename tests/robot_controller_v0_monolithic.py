import mujoco
import mujoco.viewer
import numpy as np
import time

class RobotController:
    def __init__(self, model_path):
        # Load the XML model file into MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Create the data object that holds live simulation state
        self.data = mujoco.MjData(self.model)
        
        # Print basic info so we know what loaded
        print(f"Model loaded: {self.model.nq} joints found")
        print(f"Joint positions array size: {self.data.qpos.shape}")

        self.print_robot_info()

    def print_robot_info(self):
        print("\n--- Robot Info ---")
        print(f"Total joints: {self.model.nq}")
        print(f"Total actuators: {self.model.nu}")
        print("\nJoint names and ranges:")

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            low = self.model.jnt_range[i][0]
            high = self.model.jnt_range[i][1]
            print(f"  Joint {i}: {name:20s} range: [{low:.2f}, {high:.2f}] rad")

        print("------------------\n")

    def is_safe(self, joint_index, angle):
        # read the min and max limits for this joint from the model
        # jnt_range is a 2D array where each row is [min, max] for one joint
        low = self.model.jnt_range[joint_index][0]
        high = self.model.jnt_range[joint_index][1]

        # return True only if angle sits within the safe range
        return low <= angle <= high

    def get_joint_positions(self):
        # data.qpos holds all joint angles as a numpy array
        # we return a copy so external code cant accidentally modify it
        return self.data.qpos.copy()

    def get_joint_velocities(self):
        # data.qvel holds all joint velocities
        return self.data.qvel.copy()

    def set_joint_position(self, joint_index, angle):
        # refuse the command if it would exceed joint limits
        if not self.is_safe(joint_index, angle):
            low = self.model.jnt_range[joint_index][0]
            high = self.model.jnt_range[joint_index][1]
            print(f"WARNING: Joint {joint_index} target {angle:.2f} rad "
                f"is outside safe range [{low:.2f}, {high:.2f}]")

            return

        # if safe, send the control signal
        self.data.ctrl[joint_index] = angle

    def move_to_pose(self, target_angles, steps=500):
        # target_angles is a list of 7 angles, one per joint
        # steps controls how many simulation steps to take to reach the target
        # more steps = slower and smoother movement

        # first validate every target angle before moving anything
        for i, angle in enumerate(target_angles):
            if not self.is_safe(i, angle):
                print(f"Pose rejected: joint {i} angle {angle:.2f} "
                    f"is outside safe range")

                return

        # get where the robot currently is
        start_angles = self.get_joint_positions().copy()

        # move gradually from start to target over the given number of steps
        for step in range(steps):
            # t goes from 0.0 to 1.0 as steps progress
            t = step / steps

            # interpolate between start and target for each joint
            # at t=0 you're at start, at t=1 you're at target
            interpolated = start_angles + t * (np.array(target_angles) - start_angles)

            # command each joint to its interpolated angle
            for i in range(self.model.nu):
                self.data.ctrl[i] = interpolated[i]

            # step physics and update screen
            self.step()
            self.viewer.sync()

    def get_end_effector_pos(self):
        # mujoco tracks the position of every named body in the model
        # the FR3's end effector body is named "fr3_hand"
        # mj_name2id finds its index in the model
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "fr3_hand"
        )

        # xpos holds the 3D world position of every body
        # indexed by body_id, gives us [x, y, z] in meters
        return self.data.xpos[body_id].copy()

    def get_observation(self):
        # bundle everything the agent will need into one dictionary
        # this becomes the standard way we read robot state going forward
        obs = {
            "joint_positions"  : self.get_joint_positions(),
            "joint_velocities" : self.get_joint_velocities(),
            "end_effector_pos" : self.get_end_effector_pos()
        }

        return obs

    def step(self):
        # advance the simulation by one timestep (0.002s by default)
        mujoco.mj_step(self.model, self.data)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # store viewer as instance variable so other methods can use it
            self.viewer = viewer
            print("Simulation running. Close the window to stop.")

            # move to a safe home pose on startup
            # these angles are within all 7 joints safe ranges
            home_pose = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]
            self.move_to_pose(home_pose)

            second_pose = [0.5, 1.5, 1.0, -0.2, -0.5, 1.0, 0.0]
            self.move_to_pose(second_pose)

            while viewer.is_running():
                # step physics
                self.step()

                # print full observation every 200 steps
                if int(self.data.time * 500) % 200 == 0:
                    obs = self.get_observation()
                    print(f"\nTime: {self.data.time:.2f}s")
                    print(f"Joint positions : {np.round(obs['joint_positions'], 2)}")
                    print(f"Joint velocities: {np.round(obs['joint_velocities'], 2)}")
                    print(f"End effector    : {np.round(obs['end_effector_pos'], 3)} m")

                viewer.sync()
                time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    # we will fill this path in the next step
    controller = RobotController(r"mujoco_menagerie-main\franka_fr3\fr3.xml")
    controller.run()