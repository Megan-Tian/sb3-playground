#!/usr/bin/env python3
"""
VizDoom SAC Training Script

This script trains a Soft Actor-Critic (SAC) agent to play VizDoom using the provided
DoomEnv environment with observation and action space normalization.

Features:
- Uses Stable-Baselines3 SAC implementation
- PyTorch backend
- TensorBoard logging for training metrics
- Saves trained policy for inference/deployment  
- Saves full checkpoint for resuming training
- Observation and action space normalization
- Comprehensive logging and documentation

Author: Assistant
Date: 2025
"""

import os
import argparse
from datetime import datetime
from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

# Import the environment classes (assuming they're in the same directory)
from doom import DoomEnv, NormalizedObservationWrapper, NormalizedActionWrapper, ObservationBounds


def create_normalized_env(mode: str = 'headless', 
                         obs_bounds: Optional[ObservationBounds] = None,
                         monitor_wrapper: bool = True) -> gym.Env:
    """
    Creates a normalized DoomEnv with observation and action space normalization.
    
    Args:
        mode: Environment mode ('headless', 'viz', 'img')
        obs_bounds: Observation bounds for normalization. If None, uses default bounds.
        monitor_wrapper: Whether to wrap with Monitor for logging
        
    Returns:
        Normalized DoomEnv instance
    """
    # Create base environment
    env = DoomEnv(mode=mode)
    
    # Set default observation bounds if not provided
    if obs_bounds is None:
        obs_bounds = ObservationBounds(
            min_x=-1e4,    
            max_x=1e4,    
            min_y=-1e4,  
            max_y=1e4,
            min_angle=-1e4,  
            max_angle=1e4,
            min_health=-1e4,  
            max_health=1e4
        )
    
    # Apply normalization wrappers
    env = NormalizedObservationWrapper(env, obs_bounds)
    env = NormalizedActionWrapper(env)
    
    # Wrap with Monitor for episode statistics
    if monitor_wrapper:
        env = Monitor(env)
    
    return env


def setup_directories(experiment_name: str) -> Tuple[str, str, str]:
    """
    Sets up directories for logging, models, and tensorboard.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Tuple of (log_dir, model_dir, tensorboard_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{experiment_name}_{timestamp}"
    
    log_dir = f"{base_dir}/logs"
    model_dir = f"{base_dir}/models"
    tensorboard_dir = f"{base_dir}/tensorboard"
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    return log_dir, model_dir, tensorboard_dir


def create_sac_model(env, 
                     learning_rate: float = 3e-4,
                     buffer_size: int = 1000000,
                     learning_starts: int = 100,
                     batch_size: int = 256,
                     tau: float = 0.005,
                     gamma: float = 0.99,
                     train_freq: int = 1,
                     gradient_steps: int = 1,
                     target_update_interval: int = 1,
                     tensorboard_log: Optional[str] = None,
                     device: str = 'auto',
                     verbose: int = 1) -> SAC:
    """
    Creates and configures an SAC model for VizDoom training.
    
    Args:
        env: The training environment
        learning_rate: Learning rate for all networks
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
        batch_size: Minibatch size for each gradient update
        tau: Soft update coefficient for target networks
        gamma: Discount factor
        train_freq: Update frequency (per step)
        gradient_steps: Number of gradient steps per update
        target_update_interval: Target network update frequency
        tensorboard_log: Path to tensorboard log directory
        device: Device to use ('cpu', 'cuda', 'auto')
        verbose: Verbosity level
        
    Returns:
        Configured SAC model
    """
    # Define policy network architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),  # Actor and critic network sizes
        activation_fn=torch.nn.ReLU,  # Activation function
    )
    
    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=verbose,
        seed=42  # For reproducibility
    )
    
    return model


def train_sac_agent(total_timesteps: int = 1000000,
                   experiment_name: str = "vizdoom_sac",
                   eval_freq: int = 10000,
                   n_eval_episodes: int = 5,
                   save_freq: int = 50000,
                   resume_from_checkpoint: Optional[str] = None) -> None:
    """
    Main training function for the SAC agent.
    
    Args:
        total_timesteps: Total number of training timesteps
        experiment_name: Name for the experiment (used in directory naming)
        eval_freq: Frequency of evaluation episodes
        n_eval_episodes: Number of episodes for each evaluation
        save_freq: Frequency of model checkpoints (in timesteps)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("VizDoom SAC Training Script")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Setup directories
    log_dir, model_dir, tensorboard_dir = setup_directories(experiment_name)
    print(f"Logging to: {log_dir}")
    print(f"Models saved to: {model_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = create_normalized_env(mode='headless')
    train_env = DummyVecEnv([lambda: train_env])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_normalized_env(mode='headless')
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create or load SAC model
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        model = SAC.load(resume_from_checkpoint, env=train_env, tensorboard_log=tensorboard_dir)
        print("Model loaded successfully!")
    else:
        print("\nCreating new SAC model...")
        model = create_sac_model(
            env=train_env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=1000,  # Start learning after 1000 steps
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=1,
            tensorboard_log=tensorboard_dir,
            device='auto',
            verbose=1
        )
        print("Model created successfully!")
    
    # Setup callbacks
    print("\nSetting up training callbacks...")
    
    # Evaluation callback - evaluates the agent periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="sac_vizdoom"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # Print training configuration
    print("\n" + "=" * 40)
    print("TRAINING CONFIGURATION")
    print("=" * 40)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Evaluation frequency: {eval_freq:,}")
    print(f"Checkpoint frequency: {save_freq:,}")
    print(f"Device: {model.device}")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    print("=" * 40)
    
    # Start training
    print("\nStarting training...")
    print("Monitor progress with: tensorboard --logdir=" + tensorboard_dir)
    print("Press Ctrl+C to stop training gracefully\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,  # Log every 10 episodes
            tb_log_name="SAC",
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 60)
    
    finally:
        # Save final model and checkpoint
        final_model_path = os.path.join(model_dir, "final_model")
        final_checkpoint_path = os.path.join(model_dir, "final_checkpoint.zip")
        
        print(f"\nSaving final model to: {final_model_path}")
        model.save(final_model_path)
        
        print(f"Saving final checkpoint to: {final_checkpoint_path}")
        model.save(final_checkpoint_path)
        
        # Close environments
        train_env.close()
        eval_env.close()
        
        print("\nTraining session completed!")
        print(f"Final model saved at: {final_model_path}")
        print(f"Final checkpoint saved at: {final_checkpoint_path}")
        print(f"TensorBoard logs available at: {tensorboard_dir}")


def test_trained_model(model_path: str, num_episodes: int = 5) -> None:
    """
    Tests a trained model by running it in visualization mode.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to run
    """
    print(f"\nTesting trained model: {model_path}")
    
    # Create environment for visualization
    env = create_normalized_env(mode='viz', monitor_wrapper=False)
    
    # Load the trained model
    model = SAC.load(model_path)
    
    print(f"Running {num_episodes} test episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Get action from trained model (deterministic for testing)
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
    
    env.close()
    print("Testing completed!")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Train SAC agent for VizDoom")
    
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--experiment", type=str, default="vizdoom_sac",
                       help="Experiment name (default: vizdoom_sac)")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency (default: 10,000)")
    parser.add_argument("--save-freq", type=int, default=50000,
                       help="Checkpoint save frequency (default: 50,000)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--test", type=str, default=None,
                       help="Path to model to test (skips training)")
    parser.add_argument("--test-episodes", type=int, default=5,
                       help="Number of test episodes (default: 5)")
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode - run trained model
        test_trained_model(args.test, args.test_episodes)
    else:
        # Training mode
        train_sac_agent(
            total_timesteps=args.timesteps,
            experiment_name=args.experiment,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            resume_from_checkpoint=args.resume
        )


if __name__ == "__main__":
    main()