import gymnasium as gym
import numpy as np


def inspect_env(env):
    '''
    Print a bunch of stuff about `env`
    '''
    print(f'--------------------------------------------------------------------------------')
    print(f'\tOG obs space: {env.observation_space}')
    print(f'\t\tSample obs: {env.observation_space.sample()}')
    print(f'\t\tGame variables (discrete observations): {env.unwrapped.game.get_available_game_variables()}')
    print(f'\tOG action space: {env.action_space}')
    print(f'\t\tSample action: {env.action_space.sample()}')
    print(f'\t\tAvailable buttons (actions): {env.unwrapped.game.get_available_buttons()}')        
    print(f'\tOG env spec: {env.spec}')
    print(f'--------------------------------------------------------------------------------')

class DoomLowObsEnv(gym.ObservationWrapper):
    """
    Wraps a default Doom gynasium environment to use just the low-level
    observations in the environment and discard the img.
    
    Also normalizes observations to [-1, 1], scaled from the max/min of the
    original Box observation space. 
    """
    def __init__(self, env):
        inspect_env(env)
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low = -1.0, 
            high=1.0,
            shape=env.observation_space['gamevariables'].shape,
            dtype=np.float32
        )

    def observation(self, observation):
        discrete_obs = observation['gamevariables']
        # TODO add something to confirm the OG space is bounded
        old_env_min = self.env.observation_space['gamevariables'].low
        old_env_max = self.env.observation_space['gamevariables'].high
        obs_01_scale = (discrete_obs - old_env_min) / (old_env_max - old_env_min)
        return (obs_01_scale * 2.0) - 1.0

    def render(self, mode='human'):
        '''
        Copied from scaled_auto DoomEnv
        '''
        if self.unwrapped.game.get_state() is not None:
            return self.unwrapped.game.get_state().screen_buffer
        else:
            return np.zeros((240, 320, 3))