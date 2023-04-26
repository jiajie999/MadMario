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
# env = retro.make('SuperMarioBros-Nes', 'Level1-1', use_restricted_actions=retro.Actions.FILTERED)
env=MarioImageWrapper()
env = Monitor(env, Path(logs) / datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), force=True)

# Convert to grayscale and warp frames to 84x84 (default)
# env = WarpFrame(env)
# env = FrameStack(env, num_stack=hp.FRAME_STACK)


# env = MaxSkipEnvWithRewardCoins(env, skip=hp.FRAME_SKIP)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# learning_rate = hp.LEARNING_RATE

# Got this from optHp.py
# model_params = {'n_steps': 7855/2,
#                 'gamma': 0.9043892376807653,
#                 'learning_rate': 1.049e-05,
#                 'clip_range': 0.19712472262097017,
#                 'gae_lambda': 0.9829636607957266,
#                 'batch_size': 7855/2
#                 }
model = PPO('CnnPolicy', env, verbose=1, device=device, tensorboard_log=logs)

# model = PPO('MlpPolicy', env, verbose=1, device=device, **model_params, tensorboard_log=logs)

# model.load(save_dir + '/trial_11_best_model.zip')
model.learn(total_timesteps=5000000, callback=[checkpoint_callback])
