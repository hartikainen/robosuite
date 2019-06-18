import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class InvisibleArm(Robot):
    """InvisibleArm is a witty invisible-arm robot designed by no one."""

    def __init__(self, fixed_arm=False):
        self._fixed_arm = fixed_arm
        if self._fixed_arm:
            super().__init__(xml_path_completion("robots/invisible_arm/fixed_robot.xml"))
        else:
            super().__init__(xml_path_completion("robots/invisible_arm/robot.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        if self._fixed_arm:
            return 0
        return 2

    @property
    def joints(self):
        if self._fixed_arm:
            return ()
        return ("arm_x_joint", "arm_y_joint") #, "arm_z_joint")

    @property
    def init_qpos(self):
        if self._fixed_arm:
            return np.array(())
        return np.array((0.0, 0.0))
