import sys

sys.path.append('/home/ubuntu/code/MadMario')
import os
from datetime import datetime
from pathlib import Path

import optuna as optuna
import retro
from gym.wrappers import FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import hyperparams as hp
from retroMario.mario_custom_wrapper import MaxSkipEnvWithRewardCoins
from retroMario.util import initLogDir, checkpoint_callback, get_device

logs = 'logs/'
save_dir='models'
initLogDir(logs,save_dir)

device=get_device()
def optimize_ppo(trial):
    """ 
    1. 调整nsteps的值。nsteps是PPO算法中的一个重要参数，它控制了每个训练步骤中的样本数量。
    2. 调整gamma的值。gamma是PPO算法中的一个重要参数，它控制了未来奖励的折扣因子。
    3. 调整learningrate的值。learningrate是PPO算法中的一个重要参数，它控制了模型的学习速度。
    4. 调整cliprange的值。cliprange是PPO算法中的一个重要参数，它控制了策略更新时的剪切范围。
    5. 调整gaelambda的值。gaelambda是PPO算法中的一个重要参数，它控制了GAE估计器的衰减因子。
    """
    n_steps=trial.suggest_int('n_steps', 4096, 8192)
    return {
        
        'n_steps':n_steps,
        'batch_size':n_steps,
        'gamma':  trial.suggest_loguniform('gamma', 0.995, 0.999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 5e-5, 5e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.2, 0.3),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.95, .99)
    }

def optimize_agent(trial):
    
    # try:
        model_params = optimize_ppo(trial)
        
        now=datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        env = Monitor(env, logs+f"/{now}")

        # Convert to grayscale and warp frames to 84x84 (default)
        env = WarpFrame(env)
        env = FrameStack(env, num_stack=hp.FRAME_STACK)

        # Evaluate every kth frame and repeat action
        env = MaxSkipEnvWithRewardCoins(env, skip=hp.FRAME_SKIP)
        
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')


        # learning_rate = hp.LEARNING_RATE
        # model = PPO('MlpPolicy', env, verbose=1, device=device , tensorboard_log=logs,**model_params)
        model = PPO('CnnPolicy', env, verbose=1, device=device , tensorboard_log=logs,**model_params)

        model.learn(total_timesteps=50000,callback=[checkpoint_callback])
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
        env.close()

        SAVE_PATH = os.path.join(save_dir, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        return mean_reward
    # except Exception as e:
    #      return -1000

if __name__ == '__main__':
    
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="mario-simple",
        load_if_exists=True
    )
    study.optimize(optimize_agent, n_trials=5, n_jobs=1, show_progress_bar=True)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    
    """
     Trial 24 finished with value: 603.0 and parameters: {'n_steps': 5167, 'gamma': 0.956004227649948, 'learning_rate': 4.0201928086816556e-05, 'clip_range': 0.141418947094912, 'gae_lambda': 0.960557642729578}. Best is trial 4 with value: 608.0.
    
    """