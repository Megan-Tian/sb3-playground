import gymnasium as gym
import vizdoom.gymnasium_wrapper  # This registers the VizDoom environments
import vizdoom as vzd
import numpy as np
import time
import argparse

def run_random_rollout(doom_env_path, num_steps=1000):
    """
    Run a random rollout using standard VizDoom gymnasium wrapper
    
    Args:
        doom_env_path: Path to the .cfg file
        num_steps: Number of random steps to execute
    """
    print(f"Loading VizDoom environment from: {doom_env_path}")
    
    # Create environment with custom config using Method 1
    env = gym.make('VizdoomBasic-v0', config_file_path=doom_env_path)
    
    # Access the underlying VizDoom game to configure visualization and actions
    game = env.unwrapped.game
    
    # Enable window for visualization
    game.set_window_visible(True)
    
    # Set up the actions we want: move forward, backward, turn left, turn right
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD) 
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    
    # Set up game variables for observation: position x, y, angle, health
    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    
    # Reinitialize the game with new settings
    game.init()
    
    print("Environment configured successfully!")
    print(f"Action space size: {env.action_space.n}")
    print(f"Available actions: 4 (forward, backward, turn_left, turn_right)")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Get initial game variables
    game_state = game.get_game_state()
    if game_state is not None:
        game_vars = game_state.game_variables
        print(f"Initial game variables [pos_x, pos_y, angle, health]: {game_vars}")
    
    print("Starting random rollout...")
    
    total_reward = 0
    
    for step in range(num_steps):
        # Sample random action (0-3 for our 4 actions)
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Get current game variables for display
        game_state = game.get_game_state()
        if game_state is not None:
            pos_x, pos_y, angle, health = game_state.game_variables
        else:
            pos_x = pos_y = angle = health = 0
        
        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            action_names = ['FORWARD', 'BACKWARD', 'TURN_LEFT', 'TURN_RIGHT']
            action_name = action_names[action] if action < 4 else f'ACTION_{action}'
            print(f"Step {step + 1}: Action={action_name}, "
                  f"Pos=({pos_x:.1f}, {pos_y:.1f}), Angle={angle:.1f}, Health={health:.1f}, "
                  f"Reward={reward:.2f}, Total={total_reward:.2f}")
        
        # Add small delay for visualization
        time.sleep(0.05)  # 50ms delay
        
        # Reset if episode is done
        if done or truncated:
            print(f"Episode finished at step {step + 1}")
            obs, info = env.reset()
            print("Episode reset")
    
    print(f"\nRollout completed!")
    print(f"Total steps: {num_steps}")
    print(f"Total reward: {total_reward:.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VizDoom custom environment rollout")
    parser.add_argument("DOOM_ENV_PATH", help="Path to the custom .cfg file")
    parser.add_argument("--steps", type=int, default=1000, help="Number of random steps to execute")
    
    args = parser.parse_args()
    
    try:
        run_random_rollout(args.DOOM_ENV_PATH, args.steps)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure VizDoom is properly installed and the config file path is correct")
        print("Also ensure your .cfg file is compatible with the VizDoom gymnasium wrapper")