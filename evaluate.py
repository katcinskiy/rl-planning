from env.CarEnvironment import car_environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(lambda: car_environment(display=True), n_envs=1)

episode = 0

model = PPO.load("./model_checkpoints/rl_model_5000_steps")
obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    obs, img = obs
    if done:
        print("Finished.", "reward=", reward)
        break
