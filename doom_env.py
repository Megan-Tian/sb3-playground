import random
from typing import Optional
import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper
import vizdoom

DOOM_ACTION_DIM = 4
DOOM_OBS_DIM = 4


class DoomEnv(gym.Env):
    """
    Gym wrapper for a Vizdoom environment. Comes in several modes:
        `human`: used to run human control experiments. human player.
        `headless`: used to train agents. actions are stochastic.
        `viz`: used to visualize learned agents. actions are deterministic.
        `img`: used to train deeprl agents to play from images. actions are stochastic.

    All obs (`available_game_variables`), actions (`available_buttons`), and other display
    choices must be set in the configuration file.
    """
    def __init__(self, mode="headless", config_file="deadly_corridor.cfg"):
        if mode not in ['human', 'headless', 'viz', 'img']:
            raise ValueError(f"Invalid mode: {mode}. Supported modes: ['human', 'headless', 'viz', 'img']")
        
        self.mode = mode

        # set obs and action space for each play mode
        # discrete actions: all modes 
        # low-level obs: headless, viz, human
        # image obs: img
        self.action_space = gym.spaces.Discrete(DOOM_ACTION_DIM)
        if self.mode in ['headless', 'viz', 'human']:
            self.observation_space = gym.spaces.Box(
                low=-1e4,
                high=1e4,
                shape=(DOOM_OBS_DIM,),
                dtype=np.float32
            )
        elif self.mode == 'img':
            self.observation_space = gym.spaces.Box(
                low=0,
                high=256,
                shape=(240 * 320 * 3 , ),
                dtype=np.float32
            )

        # initialize game
        self.game = vizdoom.DoomGame()
        self.game.load_config(config_file)
        self.game.init()


    def _get_obs(self):
        """
        Convert internal state to observation format.
        """
        pass

    def _get_info(self):
        """
        Compute auxiliary information for debugging. Should NOT be used by learning alg
        """
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # must call this first to seed the random number generator
        super().reset(seed=seed)

        pass

    def step(self, action: int) -> tuple[tuple, float, bool, bool, dict]:
        """
        Args:
            action: ??

        Returns:
            TODO specify type for observation, might be an nparray
            tuple: (observation, reward, terminated, truncated, info)
        """
        pass

    def render(self):
        pass

    
    def close(self):
        pass





# if __name__ == '__main__':
#     AVAILABLE_ENVS = [env for env in gym.envs.registry.keys() if "Vizdoom" in env] 
#     print(AVAILABLE_ENVS)
#     # game = VizdoomEnv("deadly_corridor.cfg", render_mode="human")

#     # Instantiate a VizDoom game instance.
#     env = gym.make("VizdoomCorridor-v0", render_mode="human")
#     # game.init()

#     print(f"env type: {type(env)}")
#     print(f"{vars(env)}")
#     print(f"{dir(env)}")
#     print(f"action space: {env.action_space}")
#     print(f'obs space: {env.observation_space}')

#     for _ in range(10):
#         done = False
#         obs, info = env.reset(seed=42)
#         while not done:
#             obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
#             done = terminated or truncated



#     # Define possible actions. Each number represents the state of a button (1=active).
#     # see sec IV.a of scaled autonomy paper
#     # actions = {
#     #     'move_forward' : [1, 0, 0, 0],
#     #     'move_backward' : [0, 1, 0, 0],
#     #     'turn_left' : [0, 0, 1, 0],
#     #     'turn_right' : [0, 0, 0, 1],
#     # }

#     # n_episodes = 10
#     # current_episode = 0
    
#     # while current_episode < n_episodes:
#     #     game.make_action(random.choice(actions.values()))

#     #     if game.is_episode_finished():
#     #         current_episode += 1
#     #         game.new_episode()

#     env.close()