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
from retroMario.util import initLogDir, checkpoint_callback

logs = 'logs/'
save_dir='models'
initLogDir(logs,save_dir)

def optimize_ppo(trial):
    """ 
    1. 调整nsteps的值。nsteps是PPO算法中的一个重要参数，它控制了每个训练步骤中的样本数量。在这个实现中，nsteps的值在2048到8192之间随机选择。您可以尝试调整这个值，看看是否能够提高模型的性能。

    2. 调整gamma的值。gamma是PPO算法中的一个重要参数，它控制了未来奖励的折扣因子。在这个实现中，gamma的值在0.8到0.9999之间随机选择。您可以尝试调整这个值，看看是否能够提高模型的性能。

    3. 调整learningrate的值。learningrate是PPO算法中的一个重要参数，它控制了模型的学习速度。在这个实现中，learningrate的值在1e-5到1e-4之间随机选择。您可以尝试调整这个值，看看是否能够提高模型的性能。

    4. 调整cliprange的值。cliprange是PPO算法中的一个重要参数，它控制了策略更新时的剪切范围。在这个实现中，cliprange的值在0.1到0.4之间随机选择。您可以尝试调整这个值，看看是否能够提高模型的性能。

    5. 调整gaelambda的值。gaelambda是PPO算法中的一个重要参数，它控制了GAE估计器的衰减因子。在这个实现中，gaelambda的值在0.8到0.99之间随机选择。您可以尝试调整这个值，看看是否能够提高模型的性能。

    """
    n_steps=trial.suggest_int('n_steps', 2048, 8192)
    return {
        
        'n_steps':n_steps,
        'batch_size':n_steps,
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, .99)
    }

def optimize_agent(trial):
    
    # try:
        model_params = optimize_ppo(trial)
        env = retro.make('SuperMarioBros-Nes', 'Level1-1', use_restricted_actions=retro.Actions.FILTERED)
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
        model = PPO('MlpPolicy', env, verbose=1, device='mps' , tensorboard_log=logs,**model_params)

        model.learn(total_timesteps=100000,callback=[checkpoint_callback])
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
    study.optimize(optimize_agent, n_trials=100, n_jobs=1, show_progress_bar=True)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    
    