import collections
from datetime import time
import gymnasium as gym
import numpy as np
from gym import spaces
from vizdoom import *
from time import sleep

DOOM_CONFIG_PATH = 'deadly_corridor.cfg'

DOOM_STATE_DIM = 4
DOOM_ACTION_DIM = 4
DOOM_ENV_SEED = 0

class DoomEnv(gym.Env):
    """
    Used to control single doom environment. Comes in several modes:
    'human': used to run human control experiments.
    'headless': used to train agents. actions are stochastic.
    'viz': used to visualize learned agents. actions are deterministic.
    'img': used to train deeprl agents to play from images. actions are stochastic.
    """
    def __init__(self, mode='headless'):
        self.idx = -1
        self.mode = mode
        if self.mode == 'human':
            self.observation_space = spaces.Box(
                low=-1e4,
                high=1e4,
                shape=(1, DOOM_STATE_DIM),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(DOOM_ACTION_DIM)
        elif self.mode == 'viz' or self.mode == 'headless':
            self.observation_space = spaces.Box(
                low=-1e4,
                high=1e4,
                shape=(DOOM_STATE_DIM,),
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0,
                high=10,
                shape=(DOOM_ACTION_DIM,),
                dtype=np.float32
            )
        elif self.mode == 'img':
            self.observation_space = spaces.Box(
                low=0,
                high=256,
                shape=(240 * 320 * 3,),
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0,
                high=10,
                shape=(DOOM_ACTION_DIM,),
                dtype=np.float32
            )

        self.game = DoomGame()
        # self.game.load_config(BASEPATH + "/ViZDoom/scenarios/deadly_corridor.cfg")
        self.game.load_config(DOOM_CONFIG_PATH)

        # button binding finagling with OG .cfg
        # this is equivalent to setting `available_buttons` to the following in the env .cfg
        # available_buttons = {MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT}
        self.game.clear_available_buttons()
        assert(self.game.get_available_buttons_size() == 0, 'should be 0 buttons available')
        self.game.add_available_button(MOVE_FORWARD)
        self.game.add_available_button(MOVE_BACKWARD)
        self.game.add_available_button(TURN_LEFT)
        self.game.add_available_button(TURN_RIGHT)
        ##################################################

        if self.mode == 'human' or self.mode == 'headless' or self.mode == 'img':
            self.game.set_window_visible(False)
        self.training = False
        self.set_state = None # Used to get around Softlearning wrappers.
        self.last_action = np.zeros(DOOM_ACTION_DIM) # Used to grab last action taken
        self.last_human_state = None

        self.game.set_seed(DOOM_ENV_SEED)
        self.game.init()

    def get_ob(self):
        if self.set_state is not None:
            return self.set_state
        state = self.game.get_state()
        if state is not None:
            if self.mode == 'img':
                return np.dstack(state.screen_buffer).reshape(-1)
            else:
                return np.array(state.game_variables)
        else:
            if self.mode == 'img':
                return np.zeros((240, 320, 3)).reshape(-1)
            else:
                return np.array([-1.] * DOOM_STATE_DIM)

    def reset(self):
        '''
        Returns blank dict for `info` in order to comply with new gymnasium
        '''
        # print(f'resetting game with seed {self.game.get_seed()} to game state {self.get_ob()}')
        self.game.new_episode()
        return self.get_ob(), dict()

    def render(self, mode='human'):
        if self.game.get_state() is not None:
            return self.game.get_state().screen_buffer
        else:
            return np.zeros((240, 320, 3))

    def done(self):
        return self.game.is_episode_finished()

    def step(self, action, immortal=False):
        '''
        Note this always returns `None` for `truncated=?` to comply with new gynmasium. OG DoomGame doesn't have
        a truncated flag.
        '''
        # Restore health to 100%
        if immortal:
            self.game.send_game_command('give health')

        # Turn action into 1-hot vector
        input_action = action
        action = [0] * DOOM_ACTION_DIM
        if not np.isclose(np.sum(input_action), 0.):
            action[np.argmax(input_action)] = 1

        # Move game forward
        old_state = self.game.get_state()
        self.last_action = action
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        state = self.game.get_state()

        # Shape Reward
        # there is also a -100 reward for dying in deadly_corridor.cfg
        if state is not None and old_state is not None and self.training:
            # Health Bonus
            reward += state.game_variables[3] / 30.
            print(f'health bonus of {state.game_variables[3] / 30.}')
            # Distance Bonus
            if state.game_variables[0] > 1100:
                reward += 5.
                print(f'distance bonus of 5.0 (case 1)')
            if state.game_variables[0] > 1200:
                reward += 5.
                print(f'distance bonus of 5.0 (case 2)')
            if state.game_variables[0] > 1250:
                reward += 5.
                print(f'distance bonus of 5.0 (case 3)')
            if state.game_variables[0] > 1275:
                reward += 5.
                print(f'distance bonus of 5.0 (case 4)')
    
        # Get observation
        if not done:
            obs = self.get_ob()
        else:
            obs = np.array([-1.] * DOOM_STATE_DIM)

        # Format return values
        if self.set_state is not None: # Need to return action to get around SAC wrapper
            return self.set_state, reward, done, None, {'action': action}
        if self.mode == 'img':
            return self.frame_buffer, reward, done, None, {'state': obs}
        else:
            return obs, reward, done, None, dict()

    def randomize_start_pos(self, agent, max_immortal_steps=150, natural_start=False):
        if natural_start:
            # Have agent play for several timesteps without taking damage
            num_immortal_steps = np.random.randint(max_immortal_steps)
            ob = self.get_ob()
            reward = 0
            done = False
            for _ in range(num_immortal_steps):
                action = agent.act(ob, reward, done)
                ob, reward, done, _ = self.step(action, immortal=True)
        else:
            # Teleport agent to random pos and deal some damage
            xpos = np.random.rand() * 1000.
            ypos = (np.random.rand() * 20.) - 10.
            health_loss = np.random.rand() * 60.
            self.game.send_game_command('warp {0} {1}'.format(xpos, ypos))
            self.game.send_game_command('take health {0}'.format(health_loss))

    def close(self):
        self.game.close()


    def get_reward_info(self):
        '''
        Returns default reward info in the environment. Disregards additional reward shaping in DoomEnv. 

        Ref: https://vizdoom.farama.org/api/python/doom_game/#reward-methods 
        '''
        return {
            'living_reward' : self.game.get_living_reward(),
            'death_penalty' : self.game.get_death_penalty(),

            # the following rewards are for Doom v1.3 and above, this repo currently on v.1.1.8
            # 'map_exit_reward' : self.game.get_map_exit_reward(),
            # 'kill_reward' : self.game.get_kill_reward(),
            # 'item_reward' : self.game.get_item_reward(),
            # 'secret_reward' : self.game.get_secret_reward(),
            # 'frag_reward' : self.game.get_frag_reward(),
            # 'hit_reward' : self.game.get_hit_reward(),
            # 'hit_taken_reward' : self.game.get_hit_taken_reward(),
            # 'damage_made_reward' : self.game.get_damage_made_reward(),
            # 'damage_taken_reward' : self.game.get_damage_taken_reward(),
            # 'health_reward' : self.game.get_health_reward(),
            # 'armor_reward' : self.game.get_armor_reward(),
        }

# register(
#     id='DoomEnv-v0',
#     entry_point='scaledauto.envs.doom:DoomEnv'
# )


ObservationBounds = collections.namedtuple('ObservationBounds', 
                                           ['min_x', 'max_x', 'min_y', 'max_y', 
                                            'min_angle', 'max_angle', 'min_health', 'max_health'])

class NormalizedObservationWrapper(gym.ObservationWrapper):
    """
    A Gym wrapper to normalize continuous observations to a [0, 1] range.
    Assumes the observation space is a Box with 4 dimensions:
    (x position, y position, angle, health)
    """
    def __init__(self, env, obs_bounds: ObservationBounds):
        super().__init__(env)
        self.min_obs = np.array([
                obs_bounds.min_x, obs_bounds.min_y, obs_bounds.min_angle, obs_bounds.min_health
            ], dtype=np.float32)
        self.max_obs = np.array([
                obs_bounds.max_x, obs_bounds.max_y, obs_bounds.max_angle, obs_bounds.max_health
            ], dtype=np.float32)
        self.observation_range = self.max_obs - self.min_obs
        # Add a small epsilon to prevent division by zero if min_obs == max_obs for any dimension
        self.observation_range[self.observation_range == 0] = 1e-6 
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32 # TODO should i normalize to [-1,1]?
        )
        print(f"Normalized observation space: {self.observation_space}")

    def observation(self, obs):
        """
        Normalizes the incoming observation.
        """
        obs = np.asarray(obs, dtype=np.float32)
        # Clip observations to the defined min/max bounds before normalizing.
        clipped_obs = np.clip(obs, self.min_obs, self.max_obs)
        normalized_to_0_1 = (clipped_obs - self.min_obs) / self.observation_range
        normalized_to_neg1_1 = (normalized_to_0_1 * 2.0) - 1.0
        return normalized_to_neg1_1




class NormalizedActionWrapper(gym.ActionWrapper):
    """
    A Gym wrapper to normalize a continuous Box action space for an agent.

    It transforms the environment's original action space [original_low, original_high]
    to a new range [-1, 1] for the agent. When the agent outputs an action
    in [-1, 1], this wrapper converts it back to the environment's original range.

    This is common for SAC where the actor network often uses tanh activation
    to produce outputs in [-1, 1].
    """
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("NormalizedActionWrapper only supports continuous Box action spaces.")

        # if env.mode not in ['viz', 'headless', 'img']:
        #     raise ValueError("NormalizedActionWrapper only supports 'viz' 'headless' 'img' modes.")

        # Store the original environment's action space bounds
        self.original_low = self.action_space.low
        self.original_high = self.action_space.high
        
        # Define the new action space that the agent will "see" (i.e., [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,  # Agent's expected output minimum
            high=1.0,   # Agent's expected output maximum
            shape=self.original_high.shape, # Keep the same shape as original
            dtype=np.float32
        )
        print(f"Original environment action space: {env.action_space}")
        print(f"Action space presented to agent (normalized): {self.action_space}")

    def action(self, action_from_agent):
        """
        Transforms an action from the agent's normalized range [-1, 1]
        to the environment's original range [original_low, original_high].

        This is the core mapping: agent's output -> environment's input.
        """
        # Ensure action is a numpy array of float32
        action_from_agent = np.asarray(action_from_agent, dtype=np.float32)

        # Clip the action to [-1, 1] to handle potential floating point errors
        # or agent outputs slightly outside the expected range.
        clipped_action = np.clip(action_from_agent, -1.0, 1.0)

        # Step 1: Scale from [-1, 1] to [0, 1]
        # This is the "scale from [0,1]" part you mentioned, but it's an intermediate step
        # for converting the agent's [-1,1] output to the environment's range.
        scaled_to_0_1 = (clipped_action + 1.0) / 2.0

        # Step 2: Scale from [0, 1] to [original_low, original_high]
        action_for_env = self.original_low + scaled_to_0_1 * (self.original_high - self.original_low)
        
        return action_for_env

    def reverse_action(self, action_from_env):
        """
        Transforms an action from the environment's original range
        to the agent's normalized range [-1, 1].
        (Useful for debugging or if you need to transform environment actions back for analysis)
        """
        # Ensure action is a numpy array of float32
        action_from_env = np.asarray(action_from_env, dtype=np.float32)

        # Step 1: Scale from [original_low, original_high] to [0, 1]
        scaled_to_0_1 = (action_from_env - self.original_low) / (self.original_high - self.original_low)
        
        # Step 2: Scale from [0, 1] to [-1, 1]
        action_for_agent = scaled_to_0_1 * 2.0 - 1.0

        return action_for_agent
    




def test_viz_continuous_actions(action='forward', print_obs=False, normalized_env=False):
    '''
    Launches `DoomEnv()` in 'viz' mode and repeats action `action` for 1k steps and 10 episodes

    action: 'forward', 'backward', 'turn_left', 'turn_right'
    '''
    action_dict = {
        'forward' : [0, 0, 0, 0],
        'backward' : [0, 1, 0, 0],
        'turn_left' : [0, 0, 1, 0],
        'turn_right' : [0, 0, 0, 1]
    }
    
    env = DoomEnv(mode='viz')

    if normalized_env:
        doom_obs_bounds = ObservationBounds(
            min_x=-1e4, max_x=1e4, # Adjust based on your environment's X range
            min_y=-1e4, max_y=1e4, # Adjust based on your environment's Y range
            min_angle=-1e4, max_angle=1e4, # Angles are typically 0-359
            min_health=-1e4, max_health=1e4 # Health is typically 0-100
        )
        env = NormalizedActionWrapper(NormalizedObservationWrapper(env, doom_obs_bounds))

        print(env.action_space)

    for _ in range(3):
        obs, _ = env.reset()
        for i in range(1000):
            act = action_dict[action]
            obs, reward, done, _, info = env.step(act)
            if print_obs:
                print(f'obs: {obs}')
                print(f'reward: {reward}')
            sleep(0.01)
            if done:
                break
    
    try:
        env.close()
        print('env closed')
    except Exception as e:
        print(f'error closing, threw error {e}')


if __name__ == "__main__":
    test_viz_continuous_actions(action='forward', print_obs=True, normalized_env=True)