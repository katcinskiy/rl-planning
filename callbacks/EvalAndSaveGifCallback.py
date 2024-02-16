from stable_baselines3.common.callbacks import BaseCallback
import imageio
import numpy as np


class EvalAndSaveGifCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, gif_path='./gifs/', verbose=0):
        super(EvalAndSaveGifCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.gif_path = gif_path
        self.n_eval_episodes = 1
        self.deterministic = True
        self.gif_prefix = 'rl_model'

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            obs = self.eval_env.reset()[0]
            frames = []
            done = False
            while not done:
                action, _ = self.model.predict(obs[np.newaxis, :], deterministic=self.deterministic)
                obs, _, done, _, _ = self.eval_env.step(action)
                frame = np.flip(np.rot90(self.eval_env.render(mode='rgb_array'), 3), axis=1)
                frames.append(frame)
            imageio.mimsave(f'{self.gif_path}{self.gif_prefix}_{self.num_timesteps}_steps.gif', frames,
                            duration=1 / 30)
        return True
