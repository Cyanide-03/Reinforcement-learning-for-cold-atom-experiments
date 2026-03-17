import numpy as np
import collections
from typing import Tuple, Dict, Optional
import os
import sys

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.artiq_bridge import ARTIQBridge
from utils.Atomcount import atom_number

class RealMOTEnvironment:
    """
    An RL environment for real-time MOT optimization using ARTIQ hardware.
    """
    
    def __init__(self, artiq_script: str = "mot_experiment.py", 
                 image_dir: str = "Dataset/real_time/",
                 detuning_range: Tuple[float, float] = (0.0, 50.0), # Physical range
                 episode_length: int = 25):
        
        self.bridge = ARTIQBridge(artiq_script=artiq_script, image_dir=image_dir)
        self.image_dir = image_dir
        self.detuning_min, self.detuning_max = detuning_range
        self.detuning_range_size = self.detuning_max - self.detuning_min
        
        self.episode_length = episode_length
        self.image_size = 50
        
        self.reset()

    def reset(self) -> Dict:
        """Resets the environment for a new real-world episode."""
        self.current_step = 0
        self.det_hist = []
        
        # Initialize image history with blank images or latest images from folder
        self.image_history = collections.deque(maxlen=4)
        
        # Optionally, pre-fill with latest images from the folder to avoid blank start
        # but for a fresh restart, blank might be safer.
        blank_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for _ in range(4):
            self.image_history.append(blank_image)
            
        print("\n--- Real Environment Reset ---")
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Executes one hardware control step.
        """
        # 1. Convert action [-1, 1] to physical detuning
        detuning_control = np.clip(action[0], -1.0, 1.0)
        # Assuming action maps to detuning range [min, max]
        # We might need to adjust signs based on user script
        scaled_detuning = self.detuning_min + (detuning_control + 1.0) / 2.0 * self.detuning_range_size
        
        # 2. Run ARTIQ experiment
        print(f"Step {self.current_step}: Setting detuning to {scaled_detuning:.2f} Γ")
        success = self.bridge.run_experiment(scaled_detuning)
        
        if not success:
            print("Warning: ARTIQ experiment failed.")
            
        # 3. Wait for new images (user said 4 images)
        # In a real experiment, this might block until the camera snaps.
        new_images = self.bridge.wait_for_new_images(count=4)
        
        # 4. Update image history
        # We take the 4 new images as the latest state
        for img in new_images:
            self.image_history.append(img)
            
        # 5. Calculate Reward
        # Based on the last image of the 4 captured
        last_image = new_images[-1]
        raw_counts = np.sum(last_image * 255.0) # Unnormalize back to 8-bit scale for formula
        
        # Use user's atom counting formula
        # Note: power, exposure etc. might be parameters
        atoms = atom_number(raw_counts, power=5, detuning=scaled_detuning, exposure=2)
        
        # For real environment, we might not have a precise Temperature per step
        # unless we do TOF. For now, we use a proxy or just Atom Number if T isn't available.
        temperature = 1.0 # Placeholder if TOF isn't performed every step
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Reward is usually final PSD, but can be incremental
        reward = (atoms / temperature) / self.episode_length if done else 0.0
        
        info = {
            'atom_number': atoms,
            'temperature': temperature,
            'detuning': scaled_detuning,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Dict:
        """Constructs observation for the agent."""
        stacked_images = np.stack(list(self.image_history), axis=0) # (4, 50, 50)
        stacked_images = np.transpose(stacked_images, (1, 2, 0)) # (50, 50, 4)
        
        normalized_time = self.current_step / self.episode_length
        # Placeholder for previous detuning if needed
        prev_det = 0.0 
        additional_inputs = np.array([normalized_time, prev_det], dtype=np.float32)
        
        return {
            'images': stacked_images,
            'additional': additional_inputs
        }
