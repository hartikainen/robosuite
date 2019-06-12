"""Gripper with 9-DoF controlling three fingers."""
import numpy as np

from robosuite.models.grippers import Gripper
from robosuite.utils.mjcf_utils import xml_path_completion


class DynamixelClawThreeFingerGripper(Gripper):
    """
    Gripper with 9 dof controlling three fingers.
    """

    def __init__(self):
        super().__init__(
            xml_path_completion("grippers/dynamixel_claw_gripper.xml"))

    def format_action(self, action):

        return action

    @property
    def init_qpos(self):
        return np.tile((0, -1.0, 1.35), 3)

    @property
    def joints(self):
        return (
            "FFJ0",
            "FFJ1",
            "FFJ2",
            "MFJ0",
            "MFJ1",
            "MFJ2",
            "THJ0",
            "THJ1",
            "THJ2",
        )

    @property
    def dof(self):
        return 9

    def contact_geoms(self):
        return (
            "palm",

            "FFLb",
            "FFL0",
            "FFL1",
            "FFL2",
            "FFTip",
            "MFLb",
            "MFL0",
            "MFL1",
            "MFL2",
            "MFTip",
            "THLb",
            "THL0",
            "THL1",
            "THL2",
            "THTip",
        )

    @property
    def visualization_sites(self):
        return ()
