import os

# base directory — everything is relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path to the Franka FR3 model XML
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mujoco_menagerie-main",
    "franka_fr3",
    "fr3.xml"
)

# simulation settings
TIMESTEP        = 0.002   # seconds per simulation step
STEPS_PER_SEC   = int(1 / TIMESTEP)   # 500

# movement settings
DEFAULT_MOVE_STEPS = 500   # steps to complete one move_to_pose

# robot poses — all angles in radians, verified within joint limits
HOME_POSE = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]

# joint safety — copied from print_robot_info output
JOINT_LIMITS = [
    (-2.74,  2.74),   # joint 0
    (-1.78,  1.78),   # joint 1
    (-2.90,  2.90),   # joint 2
    (-3.04, -0.15),   # joint 3
    (-2.81,  2.81),   # joint 4
    ( 0.54,  4.52),   # joint 5
    (-3.02,  3.02),   # joint 6
]

# end effector body name in the FR3 model XML
END_EFFECTOR_BODY = "fr3_hand"

# RL environment settings
MAX_EPISODE_STEPS = 1000    # max steps before episode ends
GOAL_THRESHOLD    = 0.05    # meters — how close counts as success
REWARD_SCALE      = 1.0     # scale factor for rewards

# tighter target bounds — closer to home pose, definitely reachable
TARGET_LOW  = [ 0.25, -0.25, 0.35]
TARGET_HIGH = [ 0.45,  0.25, 0.55]