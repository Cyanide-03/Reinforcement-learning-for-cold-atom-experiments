import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import collections
import random
from typing import Tuple, List, Optional, Dict
import cv2
from Environments.ContMOTenv import MOTEnvironmentWrapper

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state: Dict, action: np.ndarray, reward: float, 
             next_state: Dict, done: bool):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
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

class Actor(keras.Model):
    """Actor network for DDPG - outputs continuous actions"""
    
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
        """Forward pass"""
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
        
        # Scale to action space
        return self.max_action * action

class Critic(keras.Model):
    """Critic network for DDPG - estimates Q-values"""
    
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
        """Forward pass"""
        images, additional, action = inputs
        
        # Process images through CNN
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        # Concatenate state and action
        x = tf.concat([x, additional, action], axis=1)
        
        # Estimate Q-value
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q_output(x)
        
        return q_value

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration"""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.state.shape)
        self.state += dx
        return self.state


class DDPGAgent:
    """DDPG Agent for MOT control using TensorFlow"""
    
    def __init__(self, action_dim: int = 1, lr_actor: float = 1e-4, 
                 lr_critic: float = 1e-3, gamma: float = 0.99, tau: float = 0.001):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = Actor(action_dim=action_dim)
        self.critic = Critic()
        self.actor_target = Actor(action_dim=action_dim)
        self.critic_target = Critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr_critic)
        
        # Initialize networks with dummy data
        self._initialize_networks()
        
        # Initialize target networks
        self._update_target_networks(tau=1.0)
        
        # Noise for exploration
        self.noise = OUNoise(action_dim)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
    
    def _initialize_networks(self):
        """Initialize networks with dummy forward pass"""
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
        """Select action given observation"""
        # Add batch dimension
        images = np.expand_dims(observation['images'], axis=0)
        additional = np.expand_dims(observation['additional'], axis=0)
        
        # Convert to tensors
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        additional_tensor = tf.convert_to_tensor(additional, dtype=tf.float32)
        
        # Get action from actor
        action = self.actor([images_tensor, additional_tensor], training=False)
        action = action.numpy()[0]
        
        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    @tf.function
    def _train_step(self, images, additional, actions, rewards, 
                   next_images, next_additional, dones):
        """Single training step"""
        
        # Critic update
        with tf.GradientTape() as tape:
            # Target Q-values
            next_actions = self.actor_target([next_images, next_additional], training=True)
            target_q = self.critic_target([next_images, next_additional, next_actions], training=True)
            target_q = rewards + self.gamma * target_q * (1.0 - dones)
            
            # Current Q-values
            current_q = self.critic([images, additional, actions], training=True)
            
            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Actor update
        with tf.GradientTape() as tape:
            predicted_actions = self.actor([images, additional], training=True)
            actor_loss = -tf.reduce_mean(self.critic([images, additional, predicted_actions], training=True))
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        return critic_loss, actor_loss
    
    def train(self, batch_size: int = 64):
        """Train the agent"""
        if len(self.replay_buffer) < batch_size:
            return
        
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
        """Update target networks using soft update"""
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
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save_model(self, filepath: str):
        """Save trained models"""
        self.actor.save_weights(f"{filepath}_actor")
        self.critic.save_weights(f"{filepath}_critic")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        self.actor.load_weights(f"{filepath}_actor")
        self.critic.load_weights(f"{filepath}_critic")
        # Update target networks
        self._update_target_networks(tau=1.0)

def train_mot_agent(episodes: int = 10000, log_dir: str = "logs/"):
    """Main training loop with TensorBoard logging"""
    
    # Initialize environment and agent
    # Replace 'your_simulation_model' with your actual simulation
    env = MOTEnvironmentWrapper(your_simulation_model=None)  # Replace None
    agent = DDPGAgent()
    
    # TensorBoard logging
    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
    eval_summary_writer = tf.summary.create_file_writer(log_dir + '/eval')
    
    # Training parameters
    warmup_episodes = 400
    evaluation_frequency = 200
    batch_size = 64
    
    episode_rewards = []
    
    for episode in range(episodes):
        
        if episode < warmup_episodes:
            # Warmup phase with random exploration
            observation = env.reset()
            total_reward = 0
            
            for step in range(env.episode_length):
                action = np.random.uniform(-1, 1, size=(1,))
                next_observation, reward, done, info = env.step(action)
                
                agent.store_experience(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward
                
                if done:
                    break
        else:
            # Normal training episode
            observation = env.reset()
            agent.noise.reset()
            total_reward = 0
            
            for step in range(env.episode_length):
                action = agent.select_action(observation, add_noise=True)
                next_observation, reward, done, info = env.step(action)
                
                agent.store_experience(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward
                
                if done:
                    break
            
            # Train agent
            losses = agent.train(batch_size)
            
            # Log training metrics
            with train_summary_writer.as_default():
                tf.summary.scalar('episode_reward', total_reward, step=episode)
                tf.summary.scalar('atom_number', info['atom_number'], step=episode)
                tf.summary.scalar('temperature', info['temperature']*1e6, step=episode)  # in μK
                if losses:
                    tf.summary.scalar('critic_loss', losses[0], step=episode)
                    tf.summary.scalar('actor_loss', losses[1], step=episode)
        
        episode_rewards.append(total_reward)
        
        # Periodic evaluation
        if episode > 0 and episode % evaluation_frequency == 0:
            eval_rewards = []
            eval_atom_numbers = []
            eval_temperatures = []
            
            for eval_ep in range(10):
                obs = env.reset(perturbation_offset=0.0)  # No perturbation for evaluation
                eval_reward = 0
                
                for step in range(env.episode_length):
                    action = agent.select_action(obs, add_noise=False)
                    obs, reward, done, info = env.step(action)
                    eval_reward += reward
                    if done:
                        eval_atom_numbers.append(info['atom_number'])
                        eval_temperatures.append(info['temperature'])
                        break
                
                eval_rewards.append(eval_reward)
            
            # Log evaluation metrics
            with eval_summary_writer.as_default():
                tf.summary.scalar('episode_reward', np.mean(eval_rewards), step=episode)
                tf.summary.scalar('atom_number', np.mean(eval_atom_numbers), step=episode)
                tf.summary.scalar('temperature', np.mean(eval_temperatures)*1e6, step=episode)
            
            print(f"Episode {episode}: Avg Train Reward: {np.mean(episode_rewards[-100:]):.4f}, "
                  f"Avg Eval Reward: {np.mean(eval_rewards):.4f}, "
                  f"Avg Atoms: {np.mean(eval_atom_numbers):.2e}")
        
        # Save model periodically
        if episode > 0 and episode % 1000 == 0:
            agent.save_model(f"model_checkpoint_{episode}")
    
    return agent, episode_rewards

# Usage example
if __name__ == "__main__":
    # Set up TensorBoard logging
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/mot_rl_{current_time}"
    
    # Train agent
    print("Starting MOT RL training with TensorFlow...")
    print(f"Logging to: {log_dir}")
    print("Run 'tensorboard --logdir logs' to monitor training")
    
    trained_agent, rewards = train_mot_agent(episodes=5000, log_dir=log_dir)
    
    # Save final model
    trained_agent.save_model("final_mot_rl_model")
    print("Training completed. Model saved as 'final_mot_rl_model'")