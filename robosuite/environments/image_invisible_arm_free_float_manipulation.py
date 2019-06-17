import numpy as np
from robosuite.environments.invisible_arm_free_float_manipulation import (
    InvisibleArmFreeFloatManipulation)

class InvisibleArmImageFreeFloatManipulation(InvisibleArmFreeFloatManipulation):
    """
    Vision-based free float manipulation tasks. The `_get_observation` method still
    returns a dictionary with the full state, so use the _get_image_observation method
    instead for the image only as the state.
    """

    def __init__(self, image_shape, viewer_params, *args, **kwargs):
        """
        Args:

            image_shape (3-tuple): dimensions of the image observations (width, height, 
                depth/channels).

            viewer_params (dict): viewer camera settings, including the following keys:
                1. `azimuth` (float), `elevation` (float), `distance` (float), `lookat` 
                (array[float], dim x 1)
        """

        self.image_shape = image_shape
        self.viewer_params = viewer_params

        super(InvisibleArmImageFreeFloatManipulation, self).__init__(
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True, # Include the image in the super obs
            render_visual_mesh=False,
            *args, **kwargs)

    def _get_image_observation(self):
        return self._get_observation()["image"].reshape(-1)

    def _get_observation(self):
        # Super observation contains full state
        width, height = self.image_shape[:2]
        super_obs = super(InvisibleArmImageFreeFloatManipulation, self)._get_observation(
            image_width=width, image_height=height)
        return super_obs

    def viewer_setup(self): # Pass into params of env.
        self.viewer.cam.azimuth = self.viewer_params["azimuth"]
        self.viewer.cam.elevation = self.viewer_params["elevation"]
        self.viewer.cam.distance = self.viewer_params["distance"]
        self.viewer.cam.lookat[:] = self.viewer_params["lookat"]

        # FOR VICE
        # self.viewer.cam.azimuth = 90
        # self.viewer.cam.elevation = -27.7
        # self.viewer.cam.distance = 0.30
        # self.viewer.cam.lookat[:] = np.array([-2.48756381e-18, -2.48756381e-18,  7.32824139e-01])