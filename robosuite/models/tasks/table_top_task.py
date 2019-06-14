from robosuite.models.tasks import Task, UniformRandomSampler
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class TableTopTask(Task):
    """Creates MJCF model of a tabletop task.

    A tabletop task consists of one robot interacting with a variable
    number of objects placed on the tabletop. This class combines the
    robot, the table arena, and the objetcts into a single MJCF model.
    """

    def __init__(self,
                 mujoco_arena,
                 mujoco_robot,
                 mujoco_objects,
                 mujoco_visuals,
                 objects_initializer=None,
                 visuals_initializer=None):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(mujoco_visuals)

        if objects_initializer is None:
            objects_initializer = UniformRandomSampler()
        mjcfs = [x for _, x in self.mujoco_objects.items()]

        self.objects_initializer = objects_initializer
        self.objects_initializer.setup(mjcfs, self.table_top_offset, self.table_size)

        if visuals_initializer is None:
            visuals_initializer = UniformRandomSampler()
        mjcfs = [x for _, x in self.mujoco_visuals.items()]

        self.visuals_initializer = visuals_initializer
        self.visuals_initializer.setup(mjcfs, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            # obj.append(new_joint(name=obj_name + "_slide_x", type="slide", axis="1 0 0", range="-0.25 0.25"))
            # obj.append(new_joint(name=obj_name + "_slide_y", type="slide", axis="0 1 0", range="-0.25 0.25"))
            # obj.append(new_joint(name=obj_name + "_slide_z", type="slide", axis="0 0 1"))
            # obj.append(new_joint(name=obj_name + "_hinge_x", type="hinge", axis="1 0 0"))
            # obj.append(new_joint(name=obj_name + "_hinge_y", type="hinge", axis="0 1 0"))
            # obj.append(new_joint(name=obj_name + "_hinge_z", type="hinge", axis="0 0 1"))

            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(self.max_horizontal_radius,
                                             obj_mjcf.get_horizontal_radius())

    def merge_visual(self, mujoco_visuals):
        """Adds visual objects to the MJCF model."""
        self.mujoco_visuals = mujoco_visuals
        self.visuals = []  # xml manifestation

        for visual_name, visual_mjcf in mujoco_visuals.items():
            self.merge_asset(visual_mjcf)
            # Load visual
            visual = visual_mjcf.get_visual(name=visual_name, site=False)
            self.visuals.append(visual)
            self.worldbody.append(visual)

    def _place_objects(self, objects, initializer):
        pos_arr, quat_arr = initializer.sample()
        for i in range(len(objects)):
            objects[i].set("pos", array_to_string(pos_arr[i]))
            objects[i].set("quat", array_to_string(quat_arr[i]))

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations
        hit."""
        self._place_objects(self.objects, self.objects_initializer)
        self._place_objects(self.visuals, self.visuals_initializer)
