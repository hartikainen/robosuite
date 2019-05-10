import unittest

import numpy as np

import robosuite
from robosuite.models.tasks import UniformRandomSampler, ObjectPositionSampler
from robosuite.environments.invisible_arm_free_float_manipulation import (
    InvisibleArmFreeFloatManipulation)


class FixedPlacementSampler(ObjectPositionSampler):
    def __init__(self, position, quaternion):
        self._position = position
        self._quaternion = quaternion

    def sample(self):
        positions, quaternions = [], []
        for object_mjcf in self.mujoco_objects:
            bottom_offset = object_mjcf.get_bottom_offset()
            position = (self.table_top_offset - bottom_offset +
                        np.array(self._position))
            quaternion = self._quaternion
            positions.append(position)
            quaternions.append(quaternion)

        return positions, quaternions


class TestInvisibleArmFreeFloatManipulation(unittest.TestCase):
    def test_reward_for_orientation(self):
        origin_placement_initializer = FixedPlacementSampler(
            position=np.zeros(3), quaternion=np.array((1, 0, 0, 0)))
        env = robosuite.make(
            'InvisibleArmFreeFloatManipulation',
            has_renderer=False,
            use_camera_obs=False,
            placement_initializer=origin_placement_initializer)
        observation = env.reset()

        self.assertEqual(len(env.model.objects), 1)
        self.assertEqual(len(env.model.visuals), 1)

        object_body_id = list(env.object_body_ids.values())[0]
        visual_body_id = list(env.target_body_ids.values())[0]
        np.testing.assert_equal(
            env.sim.data.body_xpos[object_body_id][:2], np.zeros(2))

        action = np.zeros_like(env.action_spec[0])

        # Verify that the reward for matching position is 0
        env.sim.data.body_xpos[visual_body_id] = (
            env.sim.data.body_xpos[object_body_id])
        env.sim.data.body_xquat[visual_body_id] = (
            env.sim.data.body_xquat[object_body_id])

        observation = env._get_observation()
        np.testing.assert_equal(
            observation['screw_position'],
            observation['screw-visual_position'])
        np.testing.assert_equal(
            observation['screw_quaternion'],
            observation['screw-visual_quaternion'])
        reward = env.reward(action)

        self.assertEqual(reward, 0)

        # Verify that rewards for rotations of multiples of 2pi are 0:
        rotation_angles = np.arange(-8*np.pi, 8*np.pi+1e-6, 2*np.pi)
        expected_rewards = np.zeros_like(rotation_angles)

        for rotation_angle, expected_reward in zip(
                rotation_angles, expected_rewards):
            env.sim.data.body_xpos[visual_body_id] = (
                env.sim.data.body_xpos[object_body_id])
            env.sim.data.body_xquat[visual_body_id] = (
                [np.cos(rotation_angle / 2), 0, 0, np.sin(rotation_angle / 2)])

            observation = env._get_observation()
            reward = env.reward(action)

            np.testing.assert_allclose(reward, expected_reward, atol=1e-15)

        # Verify that rewards for rotations equal the reward of the same
        # rotation to the opposite direction:
        rotation_angles = np.linspace(0, 4*np.pi, 8)

        for rotation_angle in rotation_angles:
            env.sim.data.body_xpos[visual_body_id] = (
                env.sim.data.body_xpos[object_body_id])
            env.sim.data.body_xquat[visual_body_id] = (
                [np.cos(rotation_angle / 2), 0, 0, np.sin(rotation_angle / 2)])

            observation = env._get_observation()
            positive_angle_reward = env.reward(action)

            env.sim.data.body_xquat[visual_body_id] = (
                [np.cos(-rotation_angle / 2), 0, 0, np.sin(-rotation_angle / 2)])

            observation = env._get_observation()
            negative_angle_reward = env.reward(action)

            np.testing.assert_allclose(positive_angle_reward, negative_angle_reward)


    def test_reward_for_position(self):
        origin_placement_initializer = FixedPlacementSampler(
            position=np.zeros(3), quaternion=np.array((1, 0, 0, 0)))
        env = robosuite.make(
            'InvisibleArmFreeFloatManipulation',
            has_renderer=False,
            use_camera_obs=False,
            placement_initializer=origin_placement_initializer)
        observation = env.reset()

        self.assertEqual(len(env.model.objects), 1)
        self.assertEqual(len(env.model.visuals), 1)

        object_body_id = list(env.object_body_ids.values())[0]
        visual_body_id = list(env.target_body_ids.values())[0]
        np.testing.assert_equal(
            env.sim.data.body_xpos[object_body_id][:2], np.zeros(2))

        action = np.zeros_like(env.action_spec[0])

        # Verify that the reward for matching position is 0
        env.sim.data.body_xpos[visual_body_id] = (
            env.sim.data.body_xpos[object_body_id])
        env.sim.data.body_xquat[visual_body_id] = (
            env.sim.data.body_xquat[object_body_id])

        observation = env._get_observation()
        np.testing.assert_equal(
            observation['screw_position'],
            observation['screw-visual_position'])
        np.testing.assert_equal(
            observation['screw_quaternion'],
            observation['screw-visual_quaternion'])
        reward = env.reward(action)

        self.assertEqual(reward, 0)

        # Verify that rewards changed proportional to l2-distance:
        xs = np.arange(-1, 1+1e-10, 0.25),
        ys = np.arange(-1, 1+1e-10, 0.25),
        xys = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        for xy in xys:
            env.sim.data.body_xpos[visual_body_id] = (
                env.sim.data.body_xpos[object_body_id]
                + np.array((*xy, 0.0)))
            env.sim.data.body_xquat[visual_body_id] = (
                env.sim.data.body_xquat[object_body_id])

            reward = env.reward(action)
            expected_reward = -np.linalg.norm(xy, ord=2)
            np.testing.assert_allclose(reward, expected_reward)


if __name__ == '__main__':
    unittest.main()
