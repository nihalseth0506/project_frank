import sys
import os

# make sure project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.controller    import RobotController
from config.robot_config import MODEL_PATH


if __name__ == "__main__":
    controller = RobotController(MODEL_PATH)
    controller.run()