import numpy as np
import collections
from typing import Tuple, Dict, Optional
import os
import zmq
import sys
import base64
import cv2

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.Atomcount import atom_number

class RealMOTEnvironment:
    """
    An RL environment for real-time MOT optimization using ARTIQ hardware.
    """
    
    def __init__(self, detuning_range: Tuple[float, float] = (0.0, 50.0), episode_length: int = 25):
        
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        self.scaled_detuning = 0.0
        self.detuning_min, self.detuning_max = detuning_range
        self.detuning_range_size = self.detuning_max - self.detuning_min
        
        self.episode_length = episode_length
        self.image_size = 50
        
        self.reset()

    def reset(self) -> Dict:
        """
        Resets the environment for a new real-world episode.
        """

        self.current_step = 0
        self.det_hist = []
        
        self.image_history = collections.deque(maxlen=4)
        
        blank_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for _ in range(4):
            self.image_history.append(blank_image)
            
        print("\n--- Real Environment Reset ---")
        return self._get_observation()

    def decode_image(self, data):

        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if img is not None and img.shape != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size))
            
        if img is not None:
            img = img.astype(np.float32) / 255.0
            
        return img
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Executes one hardware control step.
        """

        detuning_control = np.clip(action[0], -1.0, 1.0)
        self.scaled_detuning = self.detuning_min + (detuning_control + 1.0) / 2.0 * self.detuning_range_size

        self.socket.send_json({
            "detuning": float(action)
        })

        response = self.socket.recv_json()
        
        imgs = [
            self.decode_image(i)
            for i in response["images"]
        ]

        
        for img in imgs:
            self.image_history.append(img)
            
        last_image = imgs[-1]
        raw_counts = np.sum(last_image * 255.0) # Unnormalize back to 8-bit scale for formula
        
        atoms = atom_number(raw_counts, power = 5, detuning = self.scaled_detuning, exposure = 2)
        np.zeros((50, 50), dtype = np.float32)
        temperature = 1.0 
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        reward = (atoms / temperature) / self.episode_length if done else 0.0
        
        info = {
            'atom_number': atoms,
            'temperature': temperature,
            'detuning': self.scaled_detuning,
        }
        
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Dict:
        """
        Constructs observation for the agent.
        """

        stacked_images = np.stack(list(self.image_history), axis=0) # (4, 50, 50)
        stacked_images = np.transpose(stacked_images, (1, 2, 0)) # (50, 50, 4)
        
        normalized_time = self.current_step / self.episode_length
        normalized_detuning = ((self.scaled_detuning / self.detuning_range_size) * 2.0) - 1.0
        additional_inputs = np.array([normalized_time, normalized_detuning], dtype=np.float32)
        
        return {
            'images': stacked_images,
            'additional': additional_inputs
        }