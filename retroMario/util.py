import os

import cv2
import numpy as np
import retro
import torch
from gym import ObservationWrapper, Wrapper, Env
from gym.spaces import Box, MultiBinary
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation, Monitor
from skimage import transform
from stable_baselines3.common.callbacks import CheckpointCallback


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


# plot ndarray img to pic
def plot_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def check_render_iamge(env):
    # plot_img(env.render())
    print("obs:====")
    print(env.observation_space.shape)


def convert(env):
    
    env = SkipFrame(env, skip=4)
    check_render_iamge(env)

    env = GrayScaleObservation(env, keep_dim=False)
    check_render_iamge(env)

    env = ResizeObservation(env, shape=84)
    check_render_iamge(env)

    env = TransformObservation(env, f=lambda x: x / 255.)
    check_render_iamge(env)

    env = FrameStack(env, num_stack=4)
    
    return env


def initLogDir(logs = 'logs/',save_dir = 'models'):
    
    os.makedirs(logs, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

# 查看retro中定义的reward函数,导入retro.data.Integrations,查看"SuperMarioBros-Nes"
def checkRewardFunc():
    from retro.data import Integrations
    env = retro.RetroEnv('SuperMarioBros-Nes')
    integration = Integrations.lookup(env.gamename)
    reward_func = integration.reward_func
    print(reward_func)
    # def reward_func(xram, info):
    #     return 0.1 * (info['screen_x'] - info['last_screen_x'])


checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/',
                                         name_prefix='ppo_')


def get_device():
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev= "mps"
    elif torch.cuda.is_available():
        dev= "cuda"
    print("Using devcie ======================= "+dev)
    return dev


"""
multi binary
{
    "Nes": {
        "lib": "fceumm",
        "ext": ["nes"],
        "keybinds": ["Z", null, "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT", "X"],
        "buttons": ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"],
        "actions": [
            [[], ["UP"], ["DOWN"]],
            [[], ["LEFT"], ["RIGHT"]],
            [[], ["A"], ["B"], ["A", "B"]]
        ]
    }
}

Discrete(36)
[]
['UP']
['DOWN']
['LEFT']
['UP', 'LEFT']
['DOWN', 'LEFT']
6 ['RIGHT']
['UP', 'RIGHT']
['DOWN', 'RIGHT']
['B']
['B', 'UP']
['B', 'DOWN']
['B', 'LEFT']
['B', 'UP', 'LEFT']
['B', 'DOWN', 'LEFT']
['B', 'RIGHT']
['B', 'UP', 'RIGHT']
['B', 'DOWN', 'RIGHT']
['A']
['UP', 'A']
['DOWN', 'A']
['LEFT', 'A']
['UP', 'LEFT', 'A']
['DOWN', 'LEFT', 'A']
24 ['RIGHT', 'A']
['UP', 'RIGHT', 'A']
['DOWN', 'RIGHT', 'A']
['B', 'A']
['B', 'UP', 'A']
['B', 'DOWN', 'A']
['B', 'LEFT', 'A']
['B', 'UP', 'LEFT', 'A']
['B', 'DOWN', 'LEFT', 'A']
['B', 'RIGHT', 'A']
['B', 'UP', 'RIGHT', 'A']
['B', 'DOWN', 'RIGHT', 'A']

"""