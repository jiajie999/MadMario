import math
import time
import collections

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymStepReturn


class MaxSkipEnvWithRewardCoins(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip
        self.last_coins = 0
        self.last_lives = 2
        self.custom_reward = 0

    def my_reward(self, reward, info):
    
        coins = info['coins']
        lives = info['lives']
        if coins > self.last_coins:
            reward += (coins - self.last_coins) * 100
        # if self.last_lives > lives:
        #     reward -= (self.last_lives - lives) * 10
    
        self.last_coins = coins
        self.last_lives = lives
        if reward>100:
            print('<<<<'+str(reward)+'>>>>>>')
            # time.sleep(5)
    
        return reward
    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            cust_reward=self.my_reward(reward,info)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward+cust_reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

 