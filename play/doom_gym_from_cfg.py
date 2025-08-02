from datetime import datetime
from gymnasium.envs.registration import register
import vizdoom as vzd
import gymnasium as gym
from vizdoom import gymnasium_wrapper


# game = vzd.DoomGame()
# game.load_config("deadly_corridor.cfg")

import inspect
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv
print(inspect.signature(VizdoomEnv.__init__))


register(
    id="deadly_corridor",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={
        "level": "./deadly_corridor.cfg", 
        "max_buttons_pressed": 1
    },
)

env = gym.make(
    "deadly_corridor", 
    render_mode="rgb_array",
)

env = gym.wrappers.RecordVideo(
    env=env, 
    video_folder="./deadly_corridor_cfg", 
    name_prefix=f"{datetime.now()}",
    episode_trigger=lambda x: True, # this records every episode
)

print(f'action space: {env.action_space}')
print(f'obs space: {env.observation_space}')

terminated = False
truncated = False
# k = 500
# env.start_video_recorder()

# running the following block w/ diff values for i (action repeated in the middle of the env)
# yields the following mappings from i -> action
# 0 do nothing
# 1 turn right
# 2 turn left
# 3 move forward
# 4 move backward
for i in [0]:
    obs, info = env.reset(seed=0)
    print(f"Testing action {i}")
    print(f'\tbefore: {obs["gamevariables"]}')

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(4)

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(i)

    print(f'\tafter: {obs["gamevariables"]}')
    if terminated or truncated:
        print(f'episode terminated or truncated')

print(f'available buttons: {env.unwrapped.game.get_available_buttons()}')
# print(f'env.max_buttons_pressed: {env.max_buttons_pressed}')
# act id : action
# 0 backware
# 1 left
# 2 right
# env.close_video_recorder()
env.close()

# print(gym.pprint_registry()) # print all the environments that can be vectorized