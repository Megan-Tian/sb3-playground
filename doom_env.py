import random
import gymnasium as gym
# import vizdoom

from vizdoom import gymnasium_wrapper

if __name__ == '__main__':
    AVAILABLE_ENVS = [env for env in gym.envs.registry.keys() if "Vizdoom" in env] 
    print(AVAILABLE_ENVS)
    # game = VizdoomEnv("deadly_corridor.cfg", render_mode="human")

    # Instantiate a VizDoom game instance.
    env = gym.make("VizdoomCorridor-v0", render_mode="human")
    # game.init()

    print(f"env type: {type(env)}")
    print(f"{vars(env)}")
    print(f"{dir(env)}")
    print(f"action space: {env.action_space}")
    print(f'obs space: {env.observation_space}')

    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated



    # Define possible actions. Each number represents the state of a button (1=active).
    # see sec IV.a of scaled autonomy paper
    # actions = {
    #     'move_forward' : [1, 0, 0, 0],
    #     'move_backward' : [0, 1, 0, 0],
    #     'turn_left' : [0, 0, 1, 0],
    #     'turn_right' : [0, 0, 0, 1],
    # }

    # n_episodes = 10
    # current_episode = 0
    
    # while current_episode < n_episodes:
    #     game.make_action(random.choice(actions.values()))

    #     if game.is_episode_finished():
    #         current_episode += 1
    #         game.new_episode()

    env.close()