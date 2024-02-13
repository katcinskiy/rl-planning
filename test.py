from env.CarEnvironment import car_environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = car_environment()

check_env(env)

# human_control = True
# episode = 0
# train_episodes = 1000
#
# observation_space = env.observation_space
# action_space = env.action_space
#
# while episode < train_episodes and env.run:
#     state = env.reset()
#     done = False
#     while not done and env.run:
#
#         action = 1
#         next_state, reward, done = env.step(action, human_control)
#         experience = state, action, reward, next_state, done
#
#         if done:
#             print('done')
#
#         state = next_state
#
#         if env.run is False:
#             break
#     episode += 1
#
# print("\nEnd of exploration phase\n")
# env.close()
