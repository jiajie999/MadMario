import math
import time
import collections

import gym
import numpy as np
import retro
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
        self.xscrollLo=0

    def my_reward(self, reward, info):
    
        coins = info['coins']
        lives = info['lives']
        xscrollLo=info['xscrollLo']
        if xscrollLo>self.xscrollLo:
            reward+=xscrollLo*10
        # if coins > self.last_coins:
        #     reward += (coins - self.last_coins)
        
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

from gym import Env
from gym.spaces import Box, MultiBinary

import cv2
class MarioImageWrapper(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(36)
        self.game = retro.make('SuperMarioBros-Nes', 'Level1-1', use_restricted_actions=retro.Actions.FILTERED)
        self.score = 0
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Preprocess frame from game
        frame_delta = obs
        #         - self.previous_frame
        #         self.previous_frame = obs
        
        # Shape reward
        
        reward = info['xscrollLo'] - self.score
        self.score = info['xscrollLo']
        
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        # Create initial variables
        self.score = 0
        
        return obs
    
    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84, 84, 1))
        return state
    
    def close(self):
        self.game.close()