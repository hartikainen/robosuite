from collections import OrderedDict
import numpy as np

from robosuite.utils import transform_utils
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.invisible_arm import InvisibleArmEnv

from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects.xml_objects import (
    CustomCubeObject, ScrewObject,
    CustomCubeVisualObject, ScrewVisualObject)
from robosuite.models.tasks import (
    FreeFloatOrientationTask,
    TableTopTask,
    UniformRandomSampler)


class InvisibleArmFreeFloatManipulation(InvisibleArmEnv):
    """
    This class corresponds to the stacking task for the InvisibleArm robot arm.
    """

    def __init__(
        self,
        gripper_type="DynamixelClawThreeFingerGripper",
        objects=None,
        visual_objects=None,
        table_full_size=(0.8, 0.8, 0.65),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        position_reward_weight=10.0,
        orientation_reward_weight=1.0,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """
        self._position_reward_weight = position_reward_weight
        self._orientation_reward_weight = orientation_reward_weight
        # initialize objects of interest
        if objects is None:
            mujoco_objects = OrderedDict((
                # ('custom-cube', CustomCubeObject()),
                ('screw', ScrewObject()),
            ))

        if visual_objects is None:
            visual_objects = OrderedDict((
                # ('custom-cube-visual', CustomCubeVisualObject()),
                ('screw-visual', ScrewVisualObject()),
            ))

        self.mujoco_objects = mujoco_objects
        self.visual_objects = visual_objects

        assert len(self.mujoco_objects) == len(self.visual_objects), (
            self.mujoco_objects, self.visual_objects)

        self.n_objects = len(self.mujoco_objects)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.3, 0.3],
                y_range=[-0.3, 0.3],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        """Loads an xml model, puts it in self.model."""
        super()._load_model()

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The InvisibleArm robot does not have a pedestal, we don't want to align pedestal with the table
        # self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        self.mujoco_robot.set_base_xpos(
            self.mujoco_arena.table_top_abs + (0, 0, 0.25))

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()
        self.model.place_visual()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        self.object_body_ids = OrderedDict([
            (key, self.sim.model.body_name2id(mujoco_object.name))
            for key, mujoco_object in self.mujoco_objects.items()
        ])
        # self.object_geom_ids = OrderedDict([
        #     (key, self.sim.model.geom_name2id(mujoco_object.name))
        #     for key, mujoco_object in self.mujoco_objects.items()
        # ])

        self.target_body_ids = OrderedDict([
            (key + "-visual", self.sim.model.body_name2id(mujoco_object.name + "-visual"))
            for key, mujoco_object in self.mujoco_objects.items()
        ])

        # self.target_geom_ids = OrderedDict([
        #     (key, self.sim.model.geom_name2id(mujoco_object.name))
        #     for key, mujoco_object in self.mujoco_objects.items()
        # ])

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()
        self.model.place_visual()

        # reset joint positions
        init_qpos = self.mujoco_robot.init_qpos.copy()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_qpos)

    def rewards(self, observations, actions):
        object_orientation_rewards = 0.0
        object_position_rewards = 0.0

        for object_name in self.mujoco_objects.keys():
            object_positions = observations["{}_position".format(object_name)]
            object_quaternions = observations[
                "{}_quaternion".format(object_name)]

            target_name = object_name + "-visual"
            target_positions = observations[
                "{}_position".format(target_name)]
            target_quaternions = observations[
                "{}_quaternion".format(target_name)]

            position_distances = np.linalg.norm(
                object_positions - target_positions, ord=2, axis=-1)
            rotation_coordinate_distances = (
                transform_utils.get_orientation_error(
                    target_quaternions, object_quaternions))
            rotation_distances = np.linalg.norm(
                rotation_coordinate_distances, ord=2, axis=-1)
            object_orientation_rewards -= rotation_distances
            object_position_rewards -= position_distances

        object_position_rewards *= self._position_reward_weight
        object_orientation_rewards *= self._orientation_reward_weight
        return object_position_rewards, object_orientation_rewards

    def reward(self, action):
        """
        Reward function for the task.

        The dense reward has five components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube
            Aligning: in [0, 0.5], encourages aligning one cube over the other
            Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The sparse reward only consists of the stacking component.
        However, the sparse reward is either 0 or 1.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        observations = OrderedDict((
            (key, value[None])
            for key, value in self._get_observation().items()
        ))
        actions = action[None]
        position_reward, orientation_reward = [
            rewards[0] for rewards in self.rewards(observations, actions)]
        reward = position_reward + orientation_reward
        return reward

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward, done, info = super(
            InvisibleArmFreeFloatManipulation, self)._post_action(action)
        observations = OrderedDict((
            (key, value[None])
            for key, value in self._get_observation().items()
        ))
        actions = action[None]
        position_reward, orientation_reward = [
            rewards[0] for rewards in self.rewards(observations, actions)]
        info.update({
            'position_reward': position_reward,
            'orientation_reward': orientation_reward,
        })
        return reward, done, info

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        observation = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                observation["image"], observation["depth"] = camera_obs
            else:
                observation["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            gripper_pose = transform_utils.pose2mat((
                observation["eef_pos"], observation["eef_quat"]))
            world_pose_in_gripper = transform_utils.pose_inv(gripper_pose)

            for object_name in self.mujoco_objects.keys():
                object_body_id = self.object_body_ids[object_name]
                object_position = self.sim.data.body_xpos[object_body_id]
                object_quaternion = transform_utils.convert_quat(
                    self.sim.data.body_xquat[object_body_id],
                    to="xyzw")

                observation.update((
                    ("{}_position".format(object_name), object_position),
                    ("{}_quaternion".format(object_name), object_quaternion),
                ))

                # get relative pose of object in gripper frame
                object_pose = transform_utils.pose2mat((
                    object_position, object_quaternion))
                relative_pose = transform_utils.pose_in_A_to_pose_in_B(
                    object_pose, world_pose_in_gripper)
                relative_position, relative_quaternion = (
                    transform_utils.mat2pose(relative_pose))

                observation.update((
                    ("{}_to_eef_pos".format(object_name),
                     relative_position),
                    ("{}_to_eef_quat".format(object_name),
                     relative_quaternion),
                ))

                target_name = object_name + "-visual"
                target_body_id = self.target_body_ids[target_name]
                target_position = self.sim.data.body_xpos[target_body_id]
                target_quaternion = transform_utils.convert_quat(
                    self.sim.data.body_xquat[target_body_id],
                    to="xyzw")

                observation.update((
                    ("{}_position".format(target_name), target_position),
                    ("{}_quaternion".format(target_name), target_quaternion),
                ))

        return observation

    def _check_contact(self):
        """Returns True if gripper is in contact with an object."""
        collision = False
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        success = False
        return success

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            pass
