# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import logging
import numpy as np
import random
import math

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class CartPoleEnv(gym.Env):
    """
    This environment simulates the classic cartpole in the pygazebo environment.
    """

    def __init__(self, max_steps=1000, x_threshold=1.0, theta_threshold=0.314):
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "cartpole.world"))

        self._agent = self._world.get_agent()
        logger.info("joint names: %s" % self._agent.get_joint_names())
        self._max_steps = max_steps
        self._x_threshold = x_threshold
        self._theta_threshold = theta_threshold
        self._world.info()

    def step(self, action):
        self._world.step(100)
        s1 = self._agent.get_joint_state("cartpole::physics::slider_to_cart")
        s2 = self._agent.get_joint_state("cartpole::physics::cart_to_pole")

        x, x_dot = s1.get_positions()[0], s1.get_velocities()[0]
        theta, theta_dot = s2.get_positions()[0], s2.get_velocities()[0]
        state = (x, x_dot, theta, theta_dot)
        logger.info("state: %f, %f, %f, %f" % state)

        done = math.fabs(x) > self._x_threshold or math.fabs(
            theta) > self._theta_threshold
        return np.array(state), 1.0, done, {}

    """
    Set cartpole states back to original
    """

    def reset(self):
        self._world.reset()


def main():
    env = CartPoleEnv()
    for _ in range(100):
        print("reset")
        env.reset()
        while True:
            obs, reward, done, info = env.step(None)
            if done:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gazebo.initialize()
    main()
