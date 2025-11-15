import numpy as np
import collections
from typing import Tuple, Dict, Optional
import tensorflow as tf
import sys

class MOTEnvironmentWrapper:
    """
    A wrapper for the MOT simulation to create a reinforcement learning environment.

    This class follows a structure similar to OpenAI Gym environments. It defines
    the state and action spaces, the reward function, and the step/reset logic
    for an episode-based interaction with an RL agent.
    """
    
    def __init__(self, Simulation_Model, image_size: int = 50, 
                 detuning_range: Tuple[float, float] = (0.0, 50)): # ! issue in detuning range
        """
        Args:
            Simulation_Model: An instance of the Simulation class.
            image_size: Size of fluorescence images  
            detuning_range: Min and max detuning values in units of Γ
        """
        self.sim_model = Simulation_Model
        self.image_size = image_size
        self.detuning_min, self.detuning_max = detuning_range
        self.detuning_range_size = self.detuning_max - self.detuning_min
        
        # Define episode parameters based on the reference paper
        self.episode_length = 25  # 25 time steps
        self.time_step_duration = 0.06  # 60ms per step

        #!!
        # Add evaluation tracking like reference 
        self.NEv = 10  # Number of evaluations
        self.evalshot_idx = 1

        # Track detuning history for logging
        self.det_hist = []

        self.reset()

    def loading(self,det):
        """
        Simulate the loading of the MOT by calling the simulation model.
        Returns:
            float: NORMALIZED atom loading rate.
        """
        return self.sim_model.predict_loading_rate(det) # ! Returns normalized atom number

    def compute_temperature(self,det):
        """
        Simulate the temperature of the MOT by calling the simulation model.
        Returns:
            float: NORMALIZED temperature.
        """
        return self.sim_model.predict_temperature(det) # ! Returns normalized Temperature

    def draw_MOT_img(self, det): 
        """
        Generate the fluorescence image using the CNN model from the simulation.
        The inputs to the image generator (atom number and detuning) are normalized.
        """
        # Normalize atom number over the episode length for stable input to the CNN
        norm_atoms = self.atom_number / self.episode_length
        # Normalize detuning to the range [-1, 0]
        norm_det = -det / self.sim_model.det_max 
        return self.sim_model.generate_image(norm_atoms, norm_det)

    #!! changes made and evaluation mode added    
    def reset(self, perturbation_offset: Optional[float] = None, evaluation_mode: bool = False) -> Dict:
        """Resets the environment to an initial state for a new episode."""
        self.current_step = 0
        self.atom_number = 0  # Normalized initial atom number
        self.temperature = 1.0 # Normalized initial temperature 
        
        # Training perturbation as described in paper
        if perturbation_offset is None:
            self.perturbation_offset = np.random.uniform(-5.0, 10.0) #unnormalized offset
        else:
            self.perturbation_offset = perturbation_offset
        
        # Initialize current detuning to a standard starting value
        self.current_detuning = -20  # Initial value like reference

        # Track evaluation mode for mid-episode perturbation change
        self.evaluation_mode = evaluation_mode

        # Clear detuning history
        self.det_hist = []
        
        # Initialize image history with 4 blank images, as the agent expects a stack of 4
        self.image_history = collections.deque(maxlen=4)
        blank_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for _ in range(4):
            self.image_history.append(blank_image)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Executes one time step in the environment.

        Args:
            action (np.ndarray): The action provided by the RL agent, in the range [-1, 1].

        Returns:
            A tuple containing (observation, reward, done, info).
        """
        # The agent outputs a normalized action in [-1, 1]
        detuning_control = action[0]  # Action from agent # [-1,1]

        # Convert the normalized action to a physical detuning value
        actual_detuning = self._convert_action_to_detuning(detuning_control) # [min, max]
        self.current_detuning = -actual_detuning #[-max,-min]
        
        # Apply a perturbation offset, which is unknown to the agent, to simulate real-world drift
        physical_detuning = self.current_detuning + self.perturbation_offset # [-max+perturb, -min+perturb]
        # tf.print(f"perturbed physical detuning: {physical_detuning}",output_stream=sys.stdout)
        # physical_detuning = tf.clip_by_value(physical_detuning, self.detuning_min, self.detuning_max) # [min,max]
        # tf.print(f"clipped physical detuning: {physical_detuning}",output_stream=sys.stdout)

        # tf.print(f"\ndetuning: {self.current_detuning}",output_stream=sys.stdout)

        # Implement mid-episode perturbation change during evaluation
        # Reference: changes offset at timestep 15 for second half of eval episodes
        if (self.evaluation_mode and self.evalshot_idx > self.NEv / 2 and self.current_step == int(np.round(self.episode_length * 3 / 5))):
            self.perturbation_offset += np.random.choice([-1, 1]) * 5.0
            self.current_detuning = self.current_detuning + self.perturbation_offset
        
        # --- Update MOT State based on the physical detuning ---
        # If the detuning is far from resonance, atoms are loaded and cooled
        if physical_detuning < -2.5:  # ! Match reference threshold
            new_atoms = self.loading(physical_detuning)
            self.atom_number += new_atoms
            self.temperature = self.compute_temperature(physical_detuning)
        else:
            # If too close to resonance, the atoms are heated out of the trap
            # ! We can add a decay rate here instead of total loss 
            self.atom_number = 0
            self.temperature = 5.0

        # Increment the step counter
        self.current_step += 1

        # Track detuning history
        self.det_hist.append(self.current_detuning)
        
        fluorescence_image = self.draw_MOT_img(physical_detuning)

        # Add the new image to the history, pushing out the oldest one
        self.image_history.append(fluorescence_image)
        
        # The episode is done if the maximum length is reached
        done = self.current_step>=self.episode_length
        
        # Reward is only given at the end of the episode, based on the final state
        reward = self._calculate_reward() if done else 0.0
        
        # Prepare an info dictionary with unnormalized, human-readable values
        atoms = self.atom_number * self.sim_model.N_max  # Unnormalize atom number
        temp = self.temperature * (self.sim_model.T_exp[-1]/0.1)
        # tf.print(f"total atoms: {atoms}, temperature: {temp*1e6} µK",output_stream=sys.stdout)
        info = {
            'atom_number': atoms,
            'temperature': temp,
            'physical_detuning': physical_detuning,
            'detuning': self.current_detuning,
            'perturbation_offset': self.perturbation_offset,
            'detuning_history': self.det_hist.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _convert_action_to_detuning(self, action: float) -> float:
        """Converts the agent's normalized action [-1, 1] to a physical detuning value."""
        # Ensure the action is within the expected range
        detuning_control = np.clip(action, -1.0, 1.0)  # Ensure [-1, 1]
        # Scale the action from [-1, 1] to [0, 1]
        detuning_control = (detuning_control + 1.0) / 2.0  # Convert to [0, 1]
        
        # Scale from [0, 1] to the physical detuning range [detuning_min, detuning_max]
        return self.detuning_min + detuning_control* self.detuning_range_size
    
    def _get_observation(self) -> Dict:
        """Constructs the observation dictionary for the agent."""
        # The primary observation is a stack of the 4 most recent images
        stacked_images = np.stack(list(self.image_history), axis=0)  # Shape: (4, 50, 50)
        # Reorder dimensions to (height, width, channels) for compatibility with TensorFlow CNNs
        stacked_images = np.transpose(stacked_images, (1, 2, 0))  # Shape: (50, 50, 4)
        
        # Additional inputs: normalized time step and placeholder for current control
        normalized_time = self.current_step / self.episode_length
        # Convert detuning back to [-1, 1] range for consistency with action space
        normalized_detuning = ((-self.current_detuning / self.detuning_range_size) * 2.0) - 1.0  # Back to [-1, 1]
        additional_inputs = np.array([normalized_time, normalized_detuning], dtype=np.float32)
        
        return {
            'images': stacked_images,
            'additional': additional_inputs
        }
    
    def _calculate_reward(self) -> float: # ! changes made problem here
        """
        Calculates the final reward for the episode.
        The reward is proportional to N/T (atom number divided by temperature),
        which represents the phase-space density, a key metric in cold atom experiments.
        """
        if self.atom_number <= 0.1:
            return 0.0

        # The reward is the ratio of final (normalized) atom number to final (normalized) temperature
        reward = self.atom_number / self.temperature / self.episode_length
        
        return reward
    
    def set_evaluation_mode(self, mode: bool):
        """Set evaluation mode for mid-episode perturbations"""
        self.evaluation_mode = mode
    
    def increment_eval_shot(self):
        """Increment evaluation shot counter"""
        self.evalshot_idx += 1
        if self.evalshot_idx > self.NEv:
            self.evalshot_idx = 1
