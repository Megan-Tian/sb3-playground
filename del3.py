import vizdoom as vzd
import numpy as np

env = vzd.DoomGame()
env.load_config("./deadly_corridor.cfg")
env.init()
env.new_episode()

# obs, info = env.reset(seed=0)
# print(f'env.reset() info: {info}')
print(f'action space: {env.get_available_buttons()}')
print(f'obs space: {env.get_available_game_variables()}')
