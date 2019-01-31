# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
from collections import OrderedDict
import gym
import gym.spaces
import logging
import numpy as np
import os
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class GoalTask(teacher.Task):
    def __init__(self, max_steps=100, goal_name="goal", distance_thresh=0.5):
        self._goal_name = goal_name
        self._distance_thresh = distance_thresh
        self._max_steps = max_steps

    def run(self, agent, world):
        agent_sentence = yield
        goal = world.get_agent(self._goal_name)
        for step in range(self._max_steps):
            loc, dir = agent.get_pose()
            goal_loc, _ = goal.get_pose()
            loc = np.array(loc)
            goal_loc = np.array(goal_loc)
            dist = np.linalg.norm(loc - goal_loc)
            if dist < self._distance_thresh:
                logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                             "dist: " + str(dist))
                agent_sentence = yield TeacherAction(
                    reward=1.0, sentence="Well done!", done=True)
            else:
                agent_sentence = yield TeacherAction()
        logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                     "dist: " + str(dist))
        yield TeacherAction(reward=-1.0, sentence="Failed", done=True)


class DiscreteSequence(gym.Space):
    def __init__(self, vocab_size, max_length):
        super()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self.dtype = np.int32
        self.shape = (max_length)


class SimpleNavigation(gym.Env):
    """
    In this environment, the agent will receive reward 1 when it is close enough to the goal.
    If it is still not close to the goal after max_steps, it will get reward -1.
    """

    def __init__(self, with_language=True):
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(),
                         "pioneer2dx_camera.world"))
        self._agent = self._world.get_agent()
        logger.info("joint names: %s" % self._agent.get_joint_names())
        self._joint_names = self._agent.get_joint_names()
        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        task_group.add_task(GoalTask())
        self._teacher.add_task_group(task_group)
        self._with_language = with_language

        # get observation dimension
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if with_language:
            self._observation_space = gym.spaces.Dict(
                image=gym.spaces.Box(
                    low=0, high=1, shape=image.shape, dtype=np.uint8),
                sentence=DiscreteSequence(256, 20))

            self._action_space = gym.spaces.Dict(
                control=gym.spaces.Box(
                    low=-0.2,
                    high=0.2,
                    shape=[len(self._joint_names)],
                    dtype=np.float32),
                sentence=DiscreteSequence(256, 20))
        else:
            self._observation_space = image = gym.spaces.Box(
                low=0, high=1, shape=image.shape, dtype=np.uint8)
            self._action_space = gym.spaces.Box(
                low=-0.2,
                high=0.2,
                shape=[len(self._joint_names)],
                dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        return -1., 1.

    def step(self, action):
        """
        Arguments
            action: a dictionary with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a string
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        controls = dict(zip(self._joint_names, controls))
        teacher_action = self._teacher.teach(sentence)
        self._agent.take_action(controls)
        self._world.step(100)
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if self._with_language:
            obs = OrderedDict(image=image, sentence=teacher_action.sentence)
        else:
            obs = image
        return (obs, teacher_action.reward, teacher_action.done, {})

    def reset(self):
        self._new_world()
        self._teacher.reset(self._agent, self._world)
        teacher_action = self._teacher.teach("")
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if self._with_language:
            obs = OrderedDict(image=image, sentence=teacher_action.sentence)
        else:
            obs = image
        return obs

    def _new_world(self):
        goal = self._world.get_model("goal")
        while True:
            loc = (random.random() * 4 - 2, random.random() * 4 - 2, 0)
            if np.linalg.norm(loc) > 0.5:
                break
        goal.set_pose((loc, (0, 0, 0)))
        self._agent.reset()


class SimpleNavigationNoLanguage(SimpleNavigation):
    def __init__(self):
        super(SimpleNavigationNoLanguage, self).__init__(with_language=False)


def main():
    env = SimpleNavigation()
    for _ in range(10000000):
        obs = env.reset()
        control = [random.random() * 0.2, random.random() * 0.2, 0]
        while True:
            obs, reward, done, info = env.step(
                dict(control=control, sentence="hello"))
            if done:
                logger.info("reward: " + str(reward) + "sent: " +
                            str(obs["sentence"]))
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gazebo.initialize()
    main()
