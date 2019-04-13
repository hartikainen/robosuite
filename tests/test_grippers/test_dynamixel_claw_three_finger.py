from robosuite.models.grippers import (
    GripperTester, DynamixelClawThreeFingerGripper)


def test_dynamixel_three_finger():
    dynamixel_claw_three_finger_tester(False)


def dynamixel_claw_three_finger_tester(render,
                                       total_iters=1,
                                       test_y=True):
    gripper = DynamixelClawThreeFingerGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.4",
        quat="0 0 1 0",
        gripper_low_pos=-0.025,
        gripper_high_pos=0.1,
        box_size=[0.03] * 3,
        box_density=100,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters,
                test_y=test_y)


if __name__ == "__main__":
    dynamixel_claw_three_finger_tester(True, 20, False)
