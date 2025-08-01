from time import sleep
from ppo_sb3_basic import IS_ENV_VEC, set_global_seed, init_env, HParamCallback, N_ENVS, SEED, inspect_env, make_vec_env
import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.ppo import PPO, MlpPolicy
from vizdoom import gymnasium_wrapper
from stable_baselines3.common.evaluation import evaluate_policy
from ppo_sb3_basic import init_env

def pickup_training():
    env = init_env()
    model = PPO.load("doom_ppo_model", env)
    model.learn(total_timesteps=20000, callback=HParamCallback(), reset_num_timesteps=True)
    model = PPO.save("doom_ppo_model_2")

def test_viz_model():
    video_folder = f"./videos"
    video_length = 300
    n_envs = 4
    vec_env = make_vec_env(init_env, n_envs=n_envs, seed=SEED)
    vec_env.render_mode = "rgb_array"
    valid_actions = [i for i in range(vec_env.action_space.n)]
    print(f'valid actions: {valid_actions}')

    model = PPO.load("./doom_ppo_model.zip", env=vec_env, force_reset=True) # env=None for inference w a trained model
    
    print(f'done loading model')

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(vec_env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=f"PPO7_{n_envs}_envs")
    
    obs = vec_env.reset() # NEED to do this reset or the video won't save

    total_reward = 0

    for _ in range(video_length + 1):
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, info = vec_env.step(action) # in a vec env these are all shape (n_envs,)
        total_reward += reward

        print(f'in vec env | \n\tobs: {obs}, \n\taction: {action}, \n\treward: {reward}, \n\tdone: {done}, \n\tinfo: {info}')

        vec_env.render("human")
        sleep(0.05)

        # if done:
        #     print(f"Episode DONE")
        #     break
    
    print(f"Total reward: {total_reward}")
    
    # Save the video
    vec_env.close()


def test_loading_saved_policy():
    env = init_env()
    env.reset()

    print(f'made env!')

    # the saved model does not contain the replay buffer
    model = PPO.load("doom_ppo_model.zip", env=env)

    #### TESTING ############################################################################################
    # Evaluate the OG policy
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=10, deterministic=True)
    print(f"model.policy mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Load the policy independently from the model
    saved_policy = MlpPolicy.load("doom_ppo_policy")
    mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)
    print(f"saved_policy mean_reward={mean_reward:.2f} +/- {std_reward}")



if __name__ == "__main__":
    print(f'MAIN')
    # pickup_training()
    test_viz_model()
    # test_loading_saved_policy()