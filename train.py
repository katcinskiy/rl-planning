from callbacks.EvalAndSaveGifCallback import EvalAndSaveGifCallback
from env.CarEnvironment import car_environment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(lambda: car_environment(), n_envs=1)
eval_env = car_environment(display=True)

TOTAL_TIMESTEPS = 1000000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./model_checkpoints/')
eval_callback = EvalAndSaveGifCallback(eval_env, eval_freq=1000)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo").learn(TOTAL_TIMESTEPS, progress_bar=True,
                                                                        callback=[checkpoint_callback, eval_callback])

