import numpy as np
import collections
from typing import Tuple, Dict, Optional

class MOTEnvironmentWrapper:
    """Wrapper for your MOT simulation environment"""
    
    def __init__(self, Simulation_Model, image_size: int = 50, 
                 detuning_range: Tuple[float, float] = (0.0, 8.25)): # ! issue in detuning range
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

        self.reset()

    def loading(self,det):
        """
        simulate the loading of the MOT
        """
        return self.sim_model.predict_loading_rate(det)

    def temperature(self,det):
        """
        simulate the temperature of the MOT
        """
        return self.sim_model.predict_temperature(det)

    def draw_MOT_img(self):
        """
        generate the fluorescence image using the CNN model
        """
        return self.sim_model.generate_image(self.atom_number, self.current_detuning)

    
    def reset(self, perturbation_offset: Optional[float] = None) -> Dict:
        """Reset environment for new episode"""
        self.current_step = 0
        self.atom_number = 0
        #!!
        self.temperature = 1.0 # Normalized initial temperature 
        
        # Training perturbation as described in paper
        if perturbation_offset is None:
            self.perturbation_offset = np.random.uniform(-0.1, 0.2) # !!
        else:
            self.perturbation_offset = perturbation_offset
        
        # !!
        #Initialize current detuning
        self.current_detuning = -20  # Initial value like reference
        
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
        actual_detuning = -self._convert_action_to_detuning(detuning_control) # [min, max]
        
        # Apply perturbation (unknown to agent)
        physical_detuning = actual_detuning + self.perturbation_offset # [min-offset,max-offset]
        physical_detuning = np.clip(physical_detuning, self.detuning_min, self.detuning_max) # [min,max]
        
        self.current_detuning = physical_detuning
        
        # Update MOT state using your simulation model
        # new_atoms, new_temperature, fluorescence_image = self._simulate_mot_step(physical_detuning)
        
        # !!
        # Update MOT state
        if physical_detuning < -2.5:  # Match reference threshold
            new_atoms = self.loading(physical_detuning)
            self.atom_number += new_atoms
            self.temperature = self.temperature(physical_detuning)
        else:
            # Too close to resonance - atoms lost (match reference behavior)
            self.atom_number = 0
            self.temperature = 5

        self.current_step += 1
        
        fluorescence_image = self.draw_MOT_img()

        # Update image history
        self.image_history.append(fluorescence_image)
        
        # Calculate reward (only at end of episode as in paper)
        done = self.current_step >= self.episode_length
        reward = self._calculate_reward() if done else 0.0
        
        # Prepare info
        info = {
            'atom_number': self.atom_number,
            'temperature': self.temperature,
            'physical_detuning': physical_detuning,
            'perturbation_offset': self.perturbation_offset
        }
        
        return self._get_observation(), reward, done, info
    
    # def _simulate_mot_step(self, detuning: float) -> Tuple[float, float, np.ndarray]:
    #     """Interface to your simulation model - adapt this to your implementation"""
    #     # This is where you call your trained simulation model
    #     # Replace this with calls to your actual simulation
        
    #     # Example interface (adapt to your model):
    #     # new_atoms = self.sim_model.predict_loading_rate(detuning)
    #     # temperature = self.sim_model.predict_temperature(detuning)  
    #     # fluorescence_image = self.sim_model.generate_image(
    #     #     self.atom_number + new_atoms, detuning
    #     # )
        
    #     # return new_atoms, temperature, fluorescence_image

    #     pass
    
    def _convert_action_to_detuning(self, action: float) -> float:
        """Convert normalized action [-1, 1] to detuning value"""
        return self.detuning_min + (action + 1) * 0.5 * self.detuning_range_size
    
    def _get_observation(self) -> Dict:
        """Get current observation for agent"""
        # Stack 4 most recent images (as in paper)
        stacked_images = np.stack(list(self.image_history), axis=0)  # Shape: (4, 50, 50)
        # Add batch dimension and reorder to (height, width, channels) for TensorFlow
        stacked_images = np.transpose(stacked_images, (1, 2, 0))  # Shape: (50, 50, 4)
        
        # Additional inputs: normalized time step and placeholder for current control
        normalized_time = self.current_step / self.episode_length
        normalized_detuning = 0.5  # Placeholder - agent learns from images
        additional_inputs = np.array([normalized_time, normalized_detuning], dtype=np.float32)
        
        return {
            'images': stacked_images,
            'additional': additional_inputs
        }
    
    def _calculate_reward(self) -> float:
        """Calculate reward R ∝ N/T as in paper"""
        if self.temperature > 0:
            reward = self.atom_number / (self.temperature * 1e6)  # Normalize temperature to μK
        else:
            reward = 0.0
        return reward / 1e8  # Normalize reward scale
