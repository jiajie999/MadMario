# -*- coding: utf-8 -*-
from pathlib import Path

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import hyperparams as hp
from retroMario.mario_custom_wrapper import MaxSkipEnvWithRewardCoins
from retroMario.util import initLogDir

logs = 'logs/'
save_dir = 'models'

env = retro.make('SuperMarioBros-Nes', 'Level1-1', use_restricted_actions=retro.Actions.FILTERED)

initLogDir(logs,save_dir)


# # Convert to grayscale and warp frames to 84x84 (default)
# env = WarpFrame(env)
# env = FrameStack(env, num_stack=hp.FRAME_STACK)
# # Evaluate every kth frame and repeat action
env = MaxSkipEnvWithRewardCoins(env, skip=hp.FRAME_SKIP)

model = PPO('MlpPolicy', env, verbose=1, device='mps', learning_rate=hp.LEARNING_RATE, tensorboard_log=logs)

# model.learn(total_timesteps=50000,  callback=[checkpoint_callback])

# model.load(Path(save_dir) / 'trial_12_best_model.zip')
model.load('trial_14_best_model.zip')
# model.load('m1_best_model.zip')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    if rewards > 10 and info['xscrollLo']>250:
        print(info)
        print(rewards)
        print("=============")
        env.render()
    if dones:
        obs = env.reset()
 