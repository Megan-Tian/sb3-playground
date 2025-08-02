import gymnasium as gym
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv
from gymnasium.envs.registration import register
from doom_env_wrappers import DoomLowObsEnv

register(
    id="deadly_corridor",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={
        "level": "./deadly_corridor.cfg", 
        "max_buttons_pressed": 1,
    },
)


class DoomEnv(gym.Env):
    '''
    Basic wrapper for registered custom Gym Vizdoom deadly_corridor environment.
    '''
    def __init__(self):
        self._env = gym.make("deadly_corridor")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.spec = self._env.spec
        # self.unwrapped = self._env.unwrapped
        self.np_random = self._env.np_random
        self.np_random = self._env.np_random
    
    def step(self, action):
        return self._env.step(action)
    
    def reset(self):
        return self._env.reset()
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()
    


if __name__ == "__main__":
    env = DoomEnv()
    env = DoomLowObsEnv(env)
    print(f'env.spec: {env.spec}')
    print(f'env.observation_space: {env.observation_space}')
    print(f'env.action_space: {env.action_space}')
