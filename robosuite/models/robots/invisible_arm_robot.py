import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class InvisibleArm(Robot):
    """InvisibleArm is a witty invisible-arm robot designed by no one."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/invisible_arm/robot.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        # return 3
        return 1

    @property
    def joints(self):
        # return ("arm_x_joint", "arm_y_joint", "arm_z_joint")
        return ("arm_rotation", )

    @property
    def init_qpos(self):
        return np.array((0.0, ))
