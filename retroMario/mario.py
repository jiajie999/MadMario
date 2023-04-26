import sys

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append('/home/ubuntu/code/MadMario')
import os
from datetime import datetime
from pathlib import Path

import retro
from gym.wrappers import FrameStack, Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import hyperparams as hp
from retroMario.mario_custom_wrapper import MaxSkipEnvWithRewardCoins, MarioImageWrapper
from retroMario.util import initLogDir, checkpoint_callback, get_device

logs = 'logs/'
save_dir = 'models'
initLogDir(logs, save_dir)

device = get_device()

env=MarioImageWrapper()
env = Monitor(env, Path(logs) / datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), force=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# learning_rate = hp.LEARNING_RATE

# Got this from optHp.py
model_params = {'n_steps': 5920,
                'gamma': 0.9043892376807653,
                'learning_rate': 1.049e-05,
                'clip_range': 0.19712472262097017,
                'gae_lambda': 0.9829636607957266,
                'batch_size': 5920
                }

model = PPO('CnnPolicy', env, verbose=1, device=device, tensorboard_log=logs,**model_params)

# model = PPO('MlpPolicy', env, verbose=1, device=device, **model_params, tensorboard_log=logs)
# model.load(save_dir + '/trial_11_best_model.zip')

checkpoint_callback2 = CheckpointCallback(save_freq=5000, save_path='./models/',
                                         name_prefix='ppo_')

model.learn(total_timesteps=500000, callback=[checkpoint_callback])
SAVE_PATH = os.path.join(save_dir, 'trial_{}_best_model'.format(1))
model.save(SAVE_PATH )