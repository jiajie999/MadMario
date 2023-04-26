# -*- coding: utf-8 -*-
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from retroMario.mario_custom_wrapper import MarioImageWrapper
from retroMario.util import initLogDir



eval_logs='logs-eval'
initLogDir(logs='logs-eval')

env = MarioImageWrapper()
env = Monitor(env, eval_logs)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('models/trial_1_best_model.zip')
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1, render=False)

print(mean_reward)

