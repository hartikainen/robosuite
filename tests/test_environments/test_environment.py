import argparse
import numpy as np
import robosuite as suite
import gym
import sac_envs
from sac_envs.envs.dclaw import (
        register_environments as register_dclaw_environments)
# register_dclaw_environments()
from softlearning.environments.gym import register_environments
register_environments()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", type=str, default="SawyerLift")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--render-mode", type=str, default="video")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create environment instance

    # env = gym.envs.make('DClaw3-ScrewV2-v0')
    if args.render_mode == "human":
        env = suite.make(
            args.environment_id,
            has_renderer=True,
            has_offscreen_renderer=True,
            # camera_name="frontview",
            use_camera_obs=True,
            rotation_only=True,
            fixed_arm=True,
            fixed_claw=False,
            initial_x_range=(0., 0.),
            initial_y_range=(0., 0.),
            target_x_range=(0., 0.),
            target_y_range=(0., 0.),
            render_visual_mesh=False,
            # image_shape=(32, 32, 3),
        )
    else:
        env = suite.make(
            args.environment_id,
            has_renderer=False,
            has_offscreen_renderer=False,
            camera_name="agentview",
            use_camera_obs=False,
        )

    # reset the environment
    import time
    for episode in range(args.num_episodes):
        observation = env.reset()
        t0 = time.time()
        for i in range(args.episode_length):
            action = np.random.uniform(-1, 1, env.dof)  # sample random action
            observation, reward, done, info = env.step(action)
            # if args.render_mode == "human":
            #     print(1 / env.control_freq - (time.time() - t0))
            #     t0 = time.time()
            env.render()  # render on display
            # print(observation['image'])

if __name__ == "__main__":
    main()
