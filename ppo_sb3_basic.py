import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
import numpy as np
import torch
import random
import gymnasium as gym
from vizdoom import gymnasium_wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


SEED = 0
DOOM_ACTION_DIM = 4
N_STEPS = 128
N_ENVS = 8
TRAIN_STEPS = 1_000_000
IS_ENV_VEC = True

LOG_DIR = f'./logs_ppo_sb3_basic_envvec_{IS_ENV_VEC}'

def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "seed" : self.model.seed,
            # TODO log more hyperparams here
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


# class DoomContActionEnv(gym.ActionWrapper):
#     """
#     [UNDER CONSTRUCTION]

#     Wraps a default Doom gynasium doom environment to take in continuous actions. 

#     Supported default environments are listed here: https://vizdoom.farama.org/environments/default/#basic
#     """
#     def __init__(self, env, shape=(DOOM_ACTION_DIM,)):
#         super().__init__(env)
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)    

#     def action(self, action):
#         pass


class DoomObsEnv(gym.ObservationWrapper):
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


# intialize env
def init_env():
    env = gym.make('VizdoomCorridor-v0')
    env = DoomObsEnv(env) # normalize observations
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01) # scale rewards down
    print(f'FINAL ENV SPEC:')
    inspect_env(env)
    return env


if __name__ == "__main__":
    set_global_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)


    if IS_ENV_VEC:
        env = make_vec_env(init_env, n_envs=N_ENVS) # vectorize env
    else:
        env = init_env()


    # initialize model
    model = PPO(
        policy='MlpPolicy', 
        env=env, 
        verbose=1, 
        # learning_rate=1e-3, 
        tensorboard_log=LOG_DIR, 
        seed=SEED
    )

    # train baby train
    model.learn(total_timesteps=TRAIN_STEPS, callback=HParamCallback())

    #### SAVING ############################################################################################
    # save the model
    model.save("doom_ppo_model")

    del model

    # the saved model does not contain the replay buffer
    model = PPO.load("doom_ppo_model")

    # Save the policy independently from the model
    # Note: if you don't save the complete model with `model.save()`
    # you cannot continue training afterward
    policy = model.policy
    policy.save("doom_ppo_policy")

    #### TESTING ############################################################################################
    env = init_env()

    # Evaluate the OG policy
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=10, deterministic=True)
    print(f"model.policy mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Load the policy independently from the model
    saved_policy = MlpPolicy.load("doom_ppo_policy")
    mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)
    print(f"saved_policy mean_reward={mean_reward:.2f} +/- {std_reward}")


    # cleanup
    env.close()
    del env
    del model
    del saved_policy