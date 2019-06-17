import os

from robosuite.environments.base import make
from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_free_float_manipulation import (
    SawyerFreeFloatManipulation)
from robosuite.environments.invisible_arm_free_float_manipulation import (
    InvisibleArmFreeFloatManipulation)
from robosuite.environments.image_invisible_arm_free_float_manipulation import (
    ImageInvisibleArmFreeFloatManipulation)
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
