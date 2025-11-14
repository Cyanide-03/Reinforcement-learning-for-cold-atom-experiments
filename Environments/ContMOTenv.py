import numpy as np
import collections
from typing import Tuple, Dict, Optional
import tensorflow as tf
import sys

class MOTEnvironmentWrapper:
    """Wrapper for your MOT simulation environment"""
    
    def __init__(self, Simulation_Model, image_size: int = 50, 
                 detuning_range: Tuple[float, float] = (0.0, 50)): # ! issue in detuning range
        """
        Args:
            Simulation_Model: Your trained simulation model
            image_size: Size of fluorescence images  
            detuning_range: Min and max detuning values in units of Γ
        """
        self.sim_model = Simulation_Model
        self.image_size = image_size
        self.detuning_min, self.detuning_max = detuning_range
        self.detuning_range_size = self.detuning_max - self.detuning_min
        
        # Episode parameters from paper
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
        simulate the loading of the MOT
        Returns NORMALIZED atom loading rate
        """
        return self.sim_model.predict_loading_rate(det) # ! Returns normalized atom number

    def compute_temperature(self,det):
        """
        simulate the temperature of the MOT
        Returns NORMALIZED temperature
        """
        return self.sim_model.predict_temperature(det) # ! Returns normalized Temperature

    def draw_MOT_img(self, det): 
        """
        generate the fluorescence image using the CNN model
        """
        norm_atoms = self.atom_number / self.episode_length
        norm_det = -(det) / self.sim_model.det_max  # Normalize detuning to [0, 1]
        return self.sim_model.generate_image(norm_atoms, norm_det)

    #!! changes made and evaluation mode added    
    def reset(self, perturbation_offset: Optional[float] = None, evaluation_mode: bool = False) -> Dict:
        """Reset environment for new episode"""
        self.current_step = 0
        self.atom_number = 0  # Normalized initial atom number
        self.temperature = 1.0 # Normalized initial temperature 
        
        # Training perturbation as described in paper
        if perturbation_offset is None:
            self.perturbation_offset = np.random.uniform(-5.0, 10.0) #unnormalized offset
        else:
            self.perturbation_offset = perturbation_offset
        
        #Initialize current detuning
        self.current_detuning = -20  # Initial value like reference

        # Track evaluation mode for mid-episode perturbation change
        self.evaluation_mode = evaluation_mode

        # Clear detuning history
        self.det_hist = []
        
        # Initialize image history (4 most recent images)
        self.image_history = collections.deque(maxlen=4)
        blank_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for _ in range(4):
            self.image_history.append(blank_image)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one time step"""
        # Convert normalized action to actual detuning
        detuning_control = action[0]  # Action from agent # [-1,1]

        #!!
        actual_detuning = self._convert_action_to_detuning(detuning_control) # [min, max]
        # tf.print(f"actual detuning: {actual_detuning}",output_stream=sys.stdout)
        self.current_detuning = -actual_detuning #[-max,-min]
        
        # Apply perturbation (unknown to agent)
        physical_detuning = self.current_detuning + self.perturbation_offset # [-max+perturb, -min+perturb]
        # tf.print(f"perturbed physical detuning: {physical_detuning}",output_stream=sys.stdout)
        # physical_detuning = tf.clip_by_value(physical_detuning, self.detuning_min, self.detuning_max) # [min,max]
        # tf.print(f"clipped physical detuning: {physical_detuning}",output_stream=sys.stdout)
        
        

        tf.print(f"\ndetuning: {self.current_detuning}",output_stream=sys.stdout)

        # Implement mid-episode perturbation change during evaluation
        # Reference: changes offset at timestep 15 for second half of eval episodes
        if (self.evaluation_mode and self.evalshot_idx > self.NEv / 2 and self.current_step == int(np.round(self.episode_length * 3 / 5))):
            self.perturbation_offset += np.random.choice([-1, 1]) * 5.0
            self.current_detuning = self.current_detuning + self.perturbation_offset
        
        
        # Update MOT state
        if physical_detuning < -2.5:  # Match reference threshold
            new_atoms = self.loading(physical_detuning)
            self.atom_number += new_atoms
            self.temperature = self.compute_temperature(physical_detuning)

        else:
            # Too close to resonance - atoms lost (match reference behavior)

            # ! We can add a decay rate here instead of total loss 
            self.atom_number = 0
            self.temperature = 5.0

        self.current_step += 1
        tf.print(f"total atoms: {self.atom_number}, temperature: {self.temperature}",output_stream=sys.stdout)

        # Track detuning history
        self.det_hist.append(self.current_detuning)
        
        fluorescence_image = self.draw_MOT_img(physical_detuning)

        # Update image history
        self.image_history.append(fluorescence_image)
        
        # Calculate reward (only at end of episode as in paper)
        done = self.current_step>=self.episode_length
        reward = self._calculate_reward() if done else 0.0
        
        # Prepare info
        atoms = self.atom_number * self.sim_model.N_max   # Unnormalize atom number
        temperature = self.temperature * (self.sim_model.T_exp[-1]/0.1)
        info = {
            'atom_number': atoms,
            'temperature': temperature,
            'physical_detuning': physical_detuning,
            'detuning': self.current_detuning,
            'perturbation_offset': self.perturbation_offset,
            'detuning_history': self.det_hist.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _convert_action_to_detuning(self, action: float) -> float:
        """Convert normalized action [-1, 1] to detuning value"""
        # Convert action [-1, 1] to detuning [detuning_min, detuning_max]

        # First convert [-1, 1] to [0, 1]
        detuning_control = np.clip(action, -1.0, 1.0)  # Ensure [-1, 1]
        detuning_control = (detuning_control + 1.0) / 2.0  # Convert to [0, 1]
        
        # Then scale to [detuning_min, detuning_max]
        return self.detuning_min + detuning_control* self.detuning_range_size
    
    def _get_observation(self) -> Dict:
        """Get current observation for agent"""
        # Stack 4 most recent images (as in paper)
        stacked_images = np.stack(list(self.image_history), axis=0)  # Shape: (4, 50, 50)
        # Add batch dimension and reorder to (height, width, channels) for TensorFlow
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
        """Calculate reward R ∝ N/T as in paper"""
        if self.atom_number <= 0.1:
            return 0.0

        # Normalize atom number (0–1)
        # norm_N = self.atom_number / self.sim_model.N_max
        
        # # Normalize temperature relative to experimental range
        # T_ref = np.mean(self.sim_model.T_exp)  # or self.sim_model.T_exp[-1]
        # norm_T = self.temperature / T_ref
        
        # # Reward ∝ N/T
        # reward = norm_N / (norm_T /1e6)
        # # reward = (self.atom_number/self.sim_model.N_max) / (self.temperature * 1e6)  # Normalize temperature to μK

        # return reward / 1e8  # Normalize reward scale
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
