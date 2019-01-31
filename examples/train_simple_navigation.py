# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import random
import social_bot
import logging
import matplotlib.pyplot as plt
import numpy as np
import PIL
from social_bot.util.replay_buffer import PrioritizedReplayBuffer
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Options(object):
    history_length = 2
    replay_buffer_size = 500000
    discount_factor = 0.99
    learning_rate = 1e-3
    learn_freq = 4
    learn_start = 10000
    exploration_start = 0.9
    exploration_end = 0.01
    f_action_feature = lambda _, action: (0.4 * (action // 6), 0.4 * (action % 6))
    # exploration linearly decrease in the first exploration_steps steps
    exploration_steps = 200000
    max_steps = int(1e8)
    batch_size = 32
    device = torch.device("cuda:1")
    prioritized_replay_alpha = 0.6
    prioritized_replay_beta0 = 0.4
    prioritized_replay_eps = 1e-3
    target_net_update_freq = 10000
    log_freq = 100
    save_freq = 1000
    action_discretize_levels = 6
    resized_image_size = (84, 84)
    model_path = '/tmp/learn_simple_navigation/agent.model'


def main():
    options = Options()
    env = gym.make("SocialBot-SimpleNavigationNoLanguage-v0")
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    image_shape = env.observation_space.shape
    agent = QAgent(
        image_shape=(image_shape[2], ) + options.resized_image_size,
        num_actions=options.action_discretize_levels**2,
        options=options)
    rewards = deque(maxlen=options.log_freq)
    steps = deque(maxlen=options.log_freq)
    total_steps = 0
    episodes = 0
    while total_steps < options.max_steps:
        reward, step = run_one_episode(env, agent, options)
        rewards.append(reward)
        steps.append(step)
        total_steps += step
        episodes += 1
        if episodes % options.log_freq == 0:
            logging.info("episodes=%s " % episodes +
                         "total_steps=%s " % total_steps +
                         "mean_reward=%s " % (sum(rewards) / len(rewards)) +
                         "mean_steps=%s " % (sum(steps) / len(steps)) +
                         "exp_rate=%s " % agent.get_exploration_rate())

        if episodes % options.save_freq == 0:
            agent.save_model(options.model_path)


def run_one_episode(env, agent, options):
    obs = env.reset()
    agent.start_new_episode()
    episode_reward = 0.
    reward = 0
    done = False
    steps = 0

    while not done:
        obs = PIL.Image.fromarray(obs).resize(options.resized_image_size,
                                              PIL.Image.ANTIALIAS)
        obs = np.transpose(obs, [2, 0, 1])
        action, q = agent.act(obs, reward)
        control = [
            0.02 * (action // options.action_discretize_levels),
            0.02 * (action % options.action_discretize_levels), 0
        ]
        new_obs, reward, done, _ = env.step(control)
        agent.learn(obs, action, reward, done)
        obs = new_obs
        episode_reward += reward
        steps += 1
    if options.log_freq == 1:
        logging.info("reward=%s" % reward + " steps=%s" % steps + " q=%s" % q)
    return episode_reward, steps


Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done"])


class QAgent(object):
    def __init__(self, image_shape, num_actions, options):
        num_image_channels = image_shape[0]
        num_input_channels = num_image_channels * (options.history_length + 1)
        if options.f_action_feature is not None:
            num_input_channels += len(
                (options.f_action_feature)(0)) * options.history_length
        self._num_actions = num_actions
        self._options = options
        self._acting_net = Network((num_input_channels, ) + image_shape[1:],
                                   num_actions).to(options.device)
        self._target_net = Network((num_input_channels, ) + image_shape[1:],
                                   num_actions).to(options.device)
        self._optimizer = optim.Adam(
            self._acting_net.parameters(), lr=options.learning_rate)
        self._episode_steps = 0
        self._total_steps = 0
        self._replay_buffer = PrioritizedReplayBuffer(
            options.replay_buffer_size,
            options.history_length,
            future_length=1)
        self._history = deque(maxlen=options.history_length)

    def act(self, obs, reward):
        eps = self.get_exploration_rate()
        if len(self._history) > 0:
            self._history[-1] = self._history[-1]._replace(reward=reward)
        if self._episode_steps < self._options.history_length:
            action = 0
            q = 0
        elif random.random() < eps:
            action = random.randint(0, self._num_actions - 1)
            q = 0
        else:
            input = self._make_input(obs, self._history)
            input = torch.from_numpy(input).float().to(self._options.device)
            self._acting_net.eval()
            with torch.no_grad():
                q_values = self._acting_net.calc_q_values(input)
            self._acting_net.train()
            q_values = q_values.cpu().data.numpy().reshape(-1)
            action = np.argmax(q_values)
            q = q_values[action]

        self._total_steps += 1
        self._episode_steps += 1
        self._history.append(Experience(obs, action, reward=0, done=False))
        return action, q

    def get_exploration_rate(self):
        p = min(1., float(self._total_steps) / self._options.exploration_steps)
        eps = (
            1 - p
        ) * self._options.exploration_start + p * self._options.exploration_end
        return eps

    def start_new_episode(self):
        self._episode_steps = 0

    def save_model(self, path):
        torch.save(self._acting_net.state_dict(), path)

    def _get_prioritized_replay_beta(self):
        p = min(1., float(self._total_steps) / self._options.max_steps)
        return (1 - p) * self._options.prioritized_replay_beta0 + p

    def learn(self, obs, action, reward, done):
        e = Experience(obs, action, reward, done)
        self._replay_buffer.add_experience(e)
        options = self._options
        if self._total_steps < options.learn_start:
            return
        if self._total_steps % options.learn_freq != 0:
            return

        inputs, actions, rewards, next_inputs, dones, is_weights, indices = \
            self._get_samples(options.batch_size)

        is_weights = is_weights.pow(self._get_prioritized_replay_beta())
        batch_size = options.batch_size
        qs = self._acting_net.calc_q_values(inputs)
        q = qs[torch.arange(batch_size, dtype=torch.long),
               actions.reshape(batch_size)]
        q = q.reshape(batch_size, 1)

        qs_next = self._acting_net.calc_q_values(next_inputs)
        qs_target = self._target_net.calc_q_values(next_inputs)
        _, a = torch.max(qs_next, dim=1)
        q_target = qs_target[torch.arange(batch_size, dtype=torch.long), a]
        q_target = q_target.reshape(batch_size, 1)
        q_target = rewards + options.discount_factor * q_target * (1 - dones)

        # minimize the loss
        q_target = q_target.detach()
        td_error = q - q_target
        loss = td_error * td_error
        priorities = abs(td_error.cpu().data.numpy()).reshape(-1)
        priorities = (priorities + options.prioritized_replay_eps
                      )**options.prioritized_replay_alpha
        self._replay_buffer.update_priority(indices, priorities)
        loss = torch.mean(loss * is_weights)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # update target network
        if self._total_steps % options.target_net_update_freq == 0:
            for target_param, param in zip(self._target_net.parameters(),
                                           self._acting_net.parameters()):
                target_param.data.copy_(param.data)

    def _make_input(self, obs, history):
        def make_action_feature(action):
            af = (self._options.f_action_feature)(action)
            af = np.array(af).astype(np.float).reshape(-1, 1, 1)
            return np.broadcast_to(af, (af.shape[0], ) + obs.shape[1:])

        scale = 2. / 255
        features = []
        for e in history:
            features.append(e.state * scale - 1)
            if self._options.f_action_feature:
                features.append(make_action_feature(e.action))
        features.append(obs * scale - 1)
        input = np.vstack(features)
        input = input.reshape((1, ) + input.shape)
        return input

    def _get_samples(self, batch_size):
        """Randomly sample a batch of experiences from memory."""

        def _make_sample(*exps):
            # inputs, actions, rewards, next_inputs, dones
            h = len(exps) - 2
            return (self._make_input(exps[h].state, exps[:h]), exps[h].action,
                    exps[h].reward,
                    self._make_input(exps[h + 1].state, exps[1:h + 1]),
                    float(exps[h].done))

        device = self._options.device
        features, indices, is_weights = self._replay_buffer.get_sample_features(
            self._options.batch_size, _make_sample)
        inputs, actions, rewards, next_inputs, dones = [
            torch.from_numpy(f) for f in features
        ]
        inputs = inputs.float().to(device)
        actions = actions.long().to(device)
        rewards = rewards.float().to(device)
        next_inputs = next_inputs.float().to(device)
        dones = dones.float().to(device)
        is_weights = torch.from_numpy(is_weights).float().to(device)
        return inputs, actions, rewards, next_inputs, dones, is_weights, indices


class Network(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Network, self).__init__()

        num_filters = (16, 32, 64)
        fc_size = (64, 64)

        self.latent_nn = nn.Sequential(
            nn.Conv2d(
                input_shape[0], num_filters[0], kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 41*41
            nn.Conv2d(
                num_filters[0],
                num_filters[1],
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 20*20
            nn.Conv2d(
                num_filters[1], num_filters[2], kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 9*9
        )
        calc_size = lambda x: (((x - 2) // 2 - 1) // 2 - 2) // 2
        latent_size = num_filters[2] * calc_size(input_shape[1]) * calc_size(
            input_shape[2])
        self.q_nn = nn.Sequential(
            nn.Linear(latent_size, fc_size[0]),
            nn.ReLU(),
            nn.Linear(fc_size[0], fc_size[1]),
            nn.ReLU(),
            nn.Linear(fc_size[1], num_actions),
        )
        self.v_nn = nn.Sequential(
            nn.Linear(latent_size, fc_size[0]),
            nn.ReLU(),
            nn.Linear(fc_size[0], fc_size[1]),
            nn.ReLU(),
            nn.Linear(fc_size[1], 1),
        )

    def calc_q_values(self, state):
        latent = self.latent_nn(state)
        latent = latent.reshape(latent.shape[0], -1)
        q_values = self.q_nn(latent)
        value = self.v_nn(latent)
        mean_q = torch.mean(q_values, dim=-1, keepdim=True)
        adjust = value - mean_q
        q_values = q_values + adjust.expand_as(q_values)
        return q_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(
        logging.FileHandler(filename='/tmp/learn_simple_navigation.log'))
    main()
