# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import logging
import numpy as np
import random
import math

from gym import spaces
import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class CartPole(gym.Env):
    """
    This environment simulates the classic cartpole in the pygazebo environment.
    """

    def __init__(self, max_steps=100, x_threshold=2.4, theta_threshold=0.314):
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "cartpole.world"))

        self._agent = self._world.get_agent()
        logger.info("joint names: %s" % self._agent.get_joint_names())
        self._max_steps = max_steps
        self._x_threshold = x_threshold
        self._theta_threshold = theta_threshold
        high = np.array([
            self._x_threshold * 2,
            np.finfo(np.float32).max, self._theta_threshold * 2,
            np.finfo(np.float32).max
        ])

        self.action_space = spaces.Box(-20., 20., shape=(1, ), dtype='float32')
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._world.info()

    def _get_state(self):
        s1 = self._agent.get_joint_state("cartpole::cartopole::slider_to_cart")
        s2 = self._agent.get_joint_state("cartpole::cartopole::cart_to_pole")

        x, x_dot = s1.get_positions()[0], s1.get_velocities()[0]
        theta, theta_dot = s2.get_positions()[0], s2.get_velocities()[0]
        state = [x, x_dot, theta, theta_dot]
        logger.info("state: %f, %f, %f, %f" % tuple(state))
        return state

    def step(self, action):
        """
          action is a single float number representing the force
          acting upon the cart
        """
        self._world.step(20)
        state = self._get_state()
        self._agent.take_action({
            "cartpole::cartopole::slider_to_cart": action
        })
        done = math.fabs(state[0]) > self._x_threshold or math.fabs(
            state[2]) > self._theta_threshold
        return state, 1.0, done, {}

    """
    Set cartpole states back to original
    """

    def reset(self):
        self._world.reset()
        return self._get_state()


def main():
    env = CartPole()
    for _ in range(100):
        print("reset")
        env.reset()
        while True:
            obs, reward, done, info = env.step(random.random() * 10)
            if done:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gazebo.initialize()
    main()
