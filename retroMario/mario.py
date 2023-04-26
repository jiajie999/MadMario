import os
from datetime import datetime
from pathlib import Path

import retro
from gym.wrappers import FrameStack, Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import hyperparams as hp
from retroMario.mario_custom_wrapper import MaxSkipEnvWithRewardCoins
from retroMario.util import initLogDir, checkpoint_callback

logs = 'logs/'
save_dir='models'
initLogDir(logs,save_dir)


env = retro.make('SuperMarioBros-Nes', 'Level1-1', use_restricted_actions=retro.Actions.FILTERED)
env = Monitor(env, Path(logs)/datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), force=True)


# Convert to grayscale and warp frames to 84x84 (default)
env = WarpFrame(env)
env = FrameStack(env, num_stack=hp.FRAME_STACK)

# Evaluate every kth frame and repeat action
env = MaxSkipEnvWithRewardCoins(env, skip=hp.FRAME_SKIP)


 
model = PPO('MlpPolicy', env, verbose=1,device='mps', learning_rate=hp.LEARNING_RATE,tensorboard_log=logs)


model.learn(total_timesteps=500000,  callback=[checkpoint_callback])

