# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import social_bot

from baselines import ppo2
from baselines import deepq
from baselines import ddpg
from baselines.run import main as M


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(
        lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    M('--alg=ppo2 --env=SocialBot-CartPole-v0'.split(' '))


def main2():
    env = gym.make('SocialBot-CartPole-v0')
    act = ddpg.learn(
        env=env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)
    print("Saving model to cartpole_model.pkl")
    act.save("~/cartpole_model.pkl")


if __name__ == '__main__':
    main()
