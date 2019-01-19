# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import logging
import numpy as np
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class CartPole(gym.Env):
    """
    This environment simulates the classic cartpole in the pygazebo environment.
    """

    def __init__(self):
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "cartpole.world"))
