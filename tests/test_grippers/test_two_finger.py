"""
Tests two finger gripper and left two finger gripper on grabbing task
"""
from robosuite.models.grippers import (
    TwoFingerGripper,
    GripperTester,
    LeftTwoFingerGripper,
)


def test_two_finger():
    two_finger_tester(False)


def two_finger_tester(render, total_iters=1, test_y=True):
    gripper = TwoFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.07,
        gripper_high_pos=0.02,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)


def test_left_two_finger():
    left_two_finger_tester(False)


def left_two_finger_tester(render, total_iters=1, test_y=True):
    gripper = LeftTwoFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=-0.07,
        gripper_high_pos=0.02,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)


if __name__ == "__main__":
    two_finger_tester(True, 20, True)
    left_two_finger_tester(True, 20, True)
