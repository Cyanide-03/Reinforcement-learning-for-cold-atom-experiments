import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import collections
import random
from typing import Tuple, List, Optional, Dict
import cv2
import os
import sys
import tqdm

# Ensure project root is on PYTHONPATH so imports like
# `from Environments.ContMOTenv import MOTEnvironmentWrapper` work
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Environments.ContMOTenv import MOTEnvironmentWrapper
from Environments.RealMOTenv import RealMOTEnvironment
from Simulation_Model.Simulation import Simulation

# # Ensure TensorFlow uses GPU if available
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ------------------------ GPU Configuration ------------------------
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Efficient allocator

    for gpu in gpus:
        # Enable memory growth (prevents TF from grabbing all memory)
        tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optional: Limit GPU memory per process (e.g., 80% of 23 GB GPU ≈ 11500 MB)
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=11500)]
        )

    print("GPU configured successfully.")
else:
    print("⚠ No GPU found. Running on CPU.")

class ReplayBuffer:
    """
    A fixed-size replay buffer to store experience tuples.
    This is a key component of off-policy RL algorithms like DDPG, allowing the
    agent to learn from a diverse set of past experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state: Dict, action: np.ndarray, reward: float, 
             next_state: Dict, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Randomly sample a batch of experiences from the buffer."""
        # random.sample is efficient for sampling from a deque
        experiences = random.sample(self.buffer, batch_size)
        
        # Extract components
        states_images = np.array([e[0]['images'] for e in experiences])
        states_additional = np.array([e[0]['additional'] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states_images = np.array([e[3]['images'] for e in experiences])
        next_states_additional = np.array([e[3]['additional'] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        
        return (states_images, states_additional, actions, rewards, 
                next_states_images, next_states_additional, dones)
    
    def __len__(self):
        return len(self.buffer)

class Actor(keras.Model): # !
    """
    The Actor network for the DDPG agent.

    It takes the current state (images and additional data) as input and outputs a
    deterministic action. Its goal is to learn a policy that maximizes the
    expected future reward.
    """
    def __init__(self, action_dim: int = 1, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # CNN layers for processing fluorescence images
        self.conv1 = layers.Conv2D(32, kernel_size=8, strides=4, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        
        self.flatten = layers.Flatten()
        
        # Fully connected layers
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(action_dim, activation='tanh')
    
    def call(self, inputs, training=None):
        """
        Defines the forward pass of the Actor network.

        Args:
            inputs: A tuple containing the image stack and additional state info.
            training: Boolean flag indicating if the model is in training mode.
        """
        images, additional = inputs
        
        # Process images through CNN
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        # Concatenate with additional inputs
        x = tf.concat([x, additional], axis=1)
        
        # Process through FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        action = self.output_layer(x)
        
        # The 'tanh' activation squashes the output to the range [-1, 1],
        # which matches the environment's expected action format.
        return action

class Critic(keras.Model):
    """
    The Critic network for the DDPG agent.

    It takes the current state and an action as input and outputs a Q-value,
    which is an estimate of the expected future reward from taking that action
    in that state. Its goal is to accurately predict the value of state-action pairs.
    """
    def __init__(self):
        super(Critic, self).__init__()
        
        # CNN layers (same as Actor)
        self.conv1 = layers.Conv2D(32, kernel_size=8, strides=4, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        
        self.flatten = layers.Flatten()
        
        # Q-value estimation layers
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.q_output = layers.Dense(1)
    
    def call(self, inputs, training=None):
        """
        Defines the forward pass of the Critic network.

        Args:
            inputs: A tuple containing the image stack, additional state info, and an action.
            training: Boolean flag indicating if the model is in training mode.
        """
        images, additional, action = inputs
        
        # Process images through CNN
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        # Concatenate the processed state representation with the action
        x = tf.concat([x, additional, action], axis=1)
        
        # Estimate Q-value
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q_output(x)
        
        return q_value

class OUNoise:
    """
    Ornstein-Uhlenbeck noise process.

    This type of noise is temporally correlated, making it suitable for exploration
    in physical systems with momentum. It helps the agent to explore the action space
    more effectively than uncorrelated noise.
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state of the noise process."""
        self.state = self.mu.copy()
    
    def sample(self):
        """Update and return a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.state.shape)
        self.state += dx
        return self.state

class DDPGAgent: # !
    """
    The Deep Deterministic Policy Gradient (DDPG) agent.

    This class encapsulates the Actor and Critic networks, their target counterparts,
    the optimizers, the replay buffer, and the training logic. It provides methods
    to select actions, store experiences, and train the networks.
    """
    def __init__(self, action_dim: int = 1, lr_actor: float = 1e-4, 
                 lr_critic: float = 1e-3, gamma: float = 0.99, tau: float = 0.001):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # --- Initialize Networks ---

        # Main networks that are trained
        self.actor = Actor(action_dim=action_dim)
        self.critic = Critic()

        # Target networks are used to stabilize training
        self.actor_target = Actor(action_dim=action_dim)
        self.critic_target = Critic()
        
        # --- Initialize Optimizers ---
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr_critic)
        
        # Build the networks by performing a dummy forward pass
        self._initialize_networks()
        
        # Copy weights from main networks to target networks
        self._update_target_networks(tau=1.0)
        
        # --- Exploration and Experience ---
        self.noise = OUNoise(action_dim)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
    
    def _initialize_networks(self):
        """
        Initializes network weights by performing a dummy forward pass.
        """

        dummy_images = tf.random.normal((1, 50, 50, 4))
        dummy_additional = tf.random.normal((1, 2))
        dummy_action = tf.random.normal((1, 1))
        
        # Initialize actor
        _ = self.actor([dummy_images, dummy_additional])
        _ = self.actor_target([dummy_images, dummy_additional])
        
        # Initialize critic  
        _ = self.critic([dummy_images, dummy_additional, dummy_action])
        _ = self.critic_target([dummy_images, dummy_additional, dummy_action])
    
    def select_action(self, observation: Dict, add_noise: bool = True) -> np.ndarray:
        """
        Selects an action based on the current observation (policy).
        """

        # Add batch dimension
        images = np.expand_dims(observation['images'], axis=0)
        additional = np.expand_dims(observation['additional'], axis=0)
        
        # Convert to tensors
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        additional_tensor = tf.convert_to_tensor(additional, dtype=tf.float32)
        
        # Get the deterministic action from the actor network
        action = self.actor([images_tensor, additional_tensor], training=False)
        action = action.numpy()[0]
        
        # Add noise for exploration during training
        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1.0, 1.0)   # ! 
        
        return action # Clipped action in [-1,1]
    
    @tf.function
    def _train_step(self, images, additional, actions, rewards, 
                   next_images, next_additional, dones):
        """
        Performs a single gradient update step for both Actor and Critic networks.
        This function is decorated with @tf.function to compile it into a high-performance
        TensorFlow graph.
        """
        
        # --- Critic Update ---
        with tf.GradientTape() as tape:
            # Predict the next action using the target actor network
            next_actions = self.actor_target([next_images, next_additional], training=True)
            # Predict the Q-value of the next state and action using the target critic
            target_q = self.critic_target([next_images, next_additional, next_actions], training=True)
            # Calculate the Bellman target: y = r + γ * Q'(s', μ'(s'))
            target_q = rewards + self.gamma * target_q * (1.0 - dones)
            
            # Get the Q-value of the current state and action from the main critic
            current_q = self.critic([images, additional, actions], training=True)
            
            # Critic loss is the Mean Squared Error between the Bellman target and the current Q-value
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        # Compute and apply gradients for the critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # --- Actor Update ---
        with tf.GradientTape() as tape:
            # Predict actions for the current states using the main actor
            predicted_actions = self.actor([images, additional], training=True)
            # Actor loss is the negative mean of the Q-values from the critic.
            # We want to update the actor in the direction that maximizes the Q-value.
            actor_loss = -tf.reduce_mean(self.critic([images, additional, predicted_actions], training=True))
        
        # Compute and apply gradients for the actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        return critic_loss, actor_loss
    
    def train(self, batch_size: int = 64):
        """
        Samples a batch from the replay buffer and performs one training step.
        """

        if len(self.replay_buffer) < batch_size:
            return # Don't train if the buffer doesn't have enough experiences
        
        # Sample batch
        (images, additional, actions, rewards, 
         next_images, next_additional, dones) = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        additional = tf.convert_to_tensor(additional, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_images = tf.convert_to_tensor(next_images, dtype=tf.float32)
        next_additional = tf.convert_to_tensor(next_additional, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Reshape rewards and dones
        rewards = tf.expand_dims(rewards, axis=1)
        dones = tf.expand_dims(dones, axis=1)
        
        # Perform training step
        critic_loss, actor_loss = self._train_step(
            images, additional, actions, rewards, 
            next_images, next_additional, dones
        )
        
        # Update target networks
        self._update_target_networks()
        
        return critic_loss.numpy(), actor_loss.numpy()
    
    def _update_target_networks(self, tau: Optional[float] = None):
        """
        Performs a "soft" update of the target networks.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        This makes the training more stable than directly copying the weights.
        """

        if tau is None:
            tau = self.tau
        
        # Update actor target
        for target_param, param in zip(self.actor_target.trainable_variables, 
                                     self.actor.trainable_variables):
            target_param.assign(target_param * (1.0 - tau) + param * tau)
        
        # Update critic target
        for target_param, param in zip(self.critic_target.trainable_variables, 
                                     self.critic.trainable_variables):
            target_param.assign(target_param * (1.0 - tau) + param * tau)
    
    def store_experience(self, state: Dict, action: np.ndarray, reward: float, 
                        next_state: Dict, done: bool):
        """
        Store experience in replay buffer
        """

        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save_model(self, filepath: str):
        """
        Save trained models
        """

        self.actor.save_weights(f"{filepath}_actor.weights.h5")
        self.critic.save_weights(f"{filepath}_critic.weights.h5")
    
    def load_model(self, filepath: str):
        """
        Load trained models
        """

        self.actor.load_weights(f"{filepath}_actor.weights.h5")
        self.critic.load_weights(f"{filepath}_critic.weights.h5")

        # Update target networks
        self._update_target_networks(tau=1.0)

def train_mot_agent(episodes: int = 10000, log_dir: str = "logs/", env_mode: str = "sim", eval_mode: bool = False):
    """
    Main training loop
    """
    
    # !! Initialize environment and agent
    if env_mode == "sim":
        print("Initializing in SIMULATION mode...")
        sim_model = Simulation()
        env = MOTEnvironmentWrapper(Simulation_Model=sim_model) 
    else:
        print("Initializing in REAL environment mode (ARTIQ)...")
        # You can customize these paths as needed
        env = RealMOTEnvironment(
            detuning_range = (0.0, 50.0),
            episode_length = 25
        )
    
    agent = DDPGAgent()

    # TensorBoard logging
    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
    eval_summary_writer = tf.summary.create_file_writer(log_dir + '/eval')
    
    # Training parameters
    # Start with random actions to populate the replay buffer before training
    warmup_episodes = 10
    evaluation_frequency = 40
    batch_size = 8
    
    tqdm_loop = tqdm.trange(episodes, desc="DDPG Training")
    
    for episode in tqdm_loop:
        
        # --- Warmup Phase ---
        if episode < warmup_episodes:
            
            tqdm_loop.set_description(f"Warmup Training")

            # During warmup, take random actions to explore the environment
            observation = env.reset()
            episode_reward = 0
            episode_detuning = []
            total_reward = 0
            # Run a full episode
            for _ in range(env.episode_length):
                action = np.random.uniform(-1, 1, size=(1,))
                next_observation, reward, done, info = env.step(action)
                
                agent.store_experience(observation, action, reward, next_observation, done)
                observation = next_observation
                episode_detuning.append(info['detuning'])
                total_reward += reward

                # tf.print(f"observation shape: {observation['images'].shape}, action: {action}, reward: {reward}",output_stream=sys.stdout)
                
                if done:
                    episode_reward = reward
                    break

        else:

            tqdm_loop.set_description(f"DDPG Training")

            # --- Training Phase ---
            observation = env.reset()
            agent.noise.reset()
            episode_reward = 0
            episode_detuning = []
            
            # Run a full episode
            for _ in range(env.episode_length):
                # Select action using the actor network + exploration noise
                action = agent.select_action(observation, add_noise = True)
                next_observation, reward, done, info = env.step(action)

                episode_detuning.append(info['detuning']);
                
                agent.store_experience(observation, action, reward, next_observation, done)
                observation = next_observation
                
                if done:
                    episode_reward = reward
                    break

            # After the episode, train the agent on a batch from the replay buffer
            losses = agent.train(batch_size)
            
            # Log training metrics to TensorBoard for monitoring
            with train_summary_writer.as_default():
                tf.summary.scalar('episode_total_reward', total_reward, step=episode)
                tf.summary.scalar('episode_reward', episode_reward, step=episode)
                tf.summary.scalar('atom_number', info['atom_number'], step=episode)
                tf.summary.scalar('temperature in μK', info['temperature']*1e6, step=episode)  # in μK
                if losses:
                    tf.summary.scalar('critic_loss', losses[0], step=episode)
                    tf.summary.scalar('actor_loss', losses[1], step=episode)

            # Use tqdm.set_postfix to display real-time training metrics
            tqdm_loop.set_postfix(
                Rew=f"{total_reward:.2f}",
                Atoms=f"{info['atom_number']:.1e}",
                Temp=f"{info['temperature']*1e6:.1f}uK",
                Det=f"{info['detuning']:.1f}Γ",
                L_C=f"{losses[0]:.3e}" if losses else "N/A"
            )

        # Log detuning sequence and final reward in separate files
        detuning_log_path = os.path.join(log_dir, "episode_detuning.txt")
        reward_log_path = os.path.join(log_dir, "episode_reward.txt")
        
        with open(detuning_log_path, "a") as log_file:
            detuning_str = " ".join(f"{float(d):.4f}" for d in episode_detuning)
            log_file.write(f"{detuning_str}\n")
            
        with open(reward_log_path, "a") as log_file:
            log_file.write(f"{float(episode_reward):.4f}\n")

        # --- Evaluation Phase ---
        # Periodic evaluation
        if eval_mode and episode > 0 and episode % evaluation_frequency == 0:
            print("\nStarting evaluation...")
            perturbation_offsets = [-0.7, -0.2, 0.0, 0.2, 0.7]  # Test robustness
            
            offset_rewards = []
            offset_atoms = []
            offset_temps = []
            
            # Run evaluation episodes with different perturbation offsets
            for eval_ep in range(4):
                obs = env.reset(perturbation_offset = perturbation_offsets[eval_ep])
                eval_reward = 0
                
                # Run a full evaluation episode
                for step in range(env.episode_length):
                    # Select action without exploration noise for evaluation
                    action = agent.select_action(obs, add_noise=False)
                    obs, reward, done, info = env.step(action)
                    
                    if done:
                        eval_reward = reward
                        offset_atoms.append(info['atom_number'])
                        offset_temps.append(info['temperature'])
                        break
                
                offset_rewards.append(eval_reward)

            # Log average evaluation metrics to TensorBoard
            with eval_summary_writer.as_default():
                tf.summary.scalar('episode_reward', np.mean(offset_rewards), step=episode)
                tf.summary.scalar('atom_number', np.mean(offset_atoms), step=episode)
                tf.summary.scalar('temperature', np.mean(offset_temps)*1e6, step=episode)
            
            # tf.print(f"=== Evaluation at Episode {episode} ===", output_stream=sys.stdout)
            # tf.print(f"Avg Train Reward (last 100): {np.mean(episode_rewards[-100:]):.4f}", output_stream=sys.stdout)
            # tf.print(f"  Offset {perturbation_offsets[eval_ep]:+.1f}Γ: ", output_stream=sys.stdout)
            # tf.print(f"Avg Eval Reward: {np.mean(offset_rewards):.4f}", output_stream=sys.stdout)
            # tf.print(f"Avg Atoms: {np.mean(offset_atoms):.2f}", output_stream=sys.stdout)
            # tf.print(f"Avg Temperature: {np.mean(offset_temps)*1e6:.2f} μK\n", output_stream=sys.stdout)
        
        # Save a model checkpoint periodically
        # if episode > 0 and episode % 1000 == 0:
        #     agent.save_model(f"model_checkpoint_{episode}")

    
    return agent, episode_reward

# Usage example
if __name__ == "__main__":
    # Set up a timestamped directory for TensorBoard logs
    import datetime
    current_time = datetime.datetime.now().strftime("Date""%Y-%m-%d"+"_Time"+ "%H-%M-%S")
    log_dir = f"logs/mot_rl_{current_time}"
    
    # Train agent
    print("Starting MOT RL training with TensorFlow...")
    print(f"Logging to: {log_dir}")
    print("Run 'tensorboard --logdir logs' to monitor training")
    
    # Choose environment mode: 'sim' or 'real'
    ENV_MODE = "real" # Change to "real" for ARTIQ training
    
    # train_mot_agent(episodes=5000, log_dir=log_dir, env_mode=ENV_MODE)
    # Run a shorter training session for demonstration purposes
    trained_agent, rewards = train_mot_agent(episodes = 30, log_dir = log_dir, env_mode = ENV_MODE, eval_mode = False)
    
    # Save final model
    # trained_agent.save_model("final_mot_rl_model")
    # print("Training completed. Model saved as 'final_mot_rl_model'")