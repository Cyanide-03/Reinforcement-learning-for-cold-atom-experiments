"""
RealMOTenv.py  —  RL Environment for Real MOT (uses RPC bridge)
================================================================
Changes from original:
  - ARTIQBridge now takes host/port instead of artiq_script/image_dir
  - reset() calls bridge.get_initial_images() for a real starting state
  - prev_det is tracked properly in the observation
"""

import numpy as np
import collections
from typing import Tuple, Dict, Optional
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.artiq_bridge import ARTIQBridge
from utils.Atomcount import atom_number


class RealMOTEnvironment:
    """
    RL environment for real-time MOT optimisation using ARTIQ hardware
    via an RPC connection to MOTController.
    """

    def __init__(self,
                 artiq_host: str = "localhost",   # IP of the ARTIQ machine
                 artiq_port: int = 3386,
                 detuning_range: Tuple[float, float] = (0.0, 50.0),
                 episode_length: int = 25):

        # Connect to the RPC server once; keep alive for the entire training run
        self.bridge = ARTIQBridge(host=artiq_host, port=artiq_port)

        self.detuning_min, self.detuning_max = detuning_range
        self.detuning_range_size = self.detuning_max - self.detuning_min
        self.episode_length = episode_length
        self.image_size = 50

        self._prev_detuning_normalized = 0.0  # tracks last action for observation
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> Dict:
        """Reset for a new episode. Uses real images from hardware as start state."""
        self.current_step = 0
        self._prev_detuning_normalized = 0.0
        self.image_history = collections.deque(maxlen=4)

        # Try to get real images; fall back to blank if server has none yet
        initial_images = self.bridge.get_initial_images()
        for img in initial_images:
            self.image_history.append(img)

        print("\n--- Real Environment Reset ---")
        return self._get_observation()

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one hardware control step via RPC."""

        # 1. Action [-1, 1]  →  physical detuning [min, max]
        detuning_control = float(np.clip(action[0], -1.0, 1.0))
        scaled_detuning = (
            self.detuning_min
            + (detuning_control + 1.0) / 2.0 * self.detuning_range_size
        )
        self._prev_detuning_normalized = detuning_control

        # 2. Run experiment on hardware (RPC call — blocks until images arrive)
        print(f"Step {self.current_step}: detuning → {scaled_detuning:.2f} Γ")
        success = self.bridge.run_experiment(scaled_detuning)
        if not success:
            print("WARNING: ARTIQ experiment reported failure.")

        # 3. Retrieve images (already cached by run_experiment)
        new_images = self.bridge.wait_for_new_images(count=4)
        for img in new_images:
            self.image_history.append(img)

        # 4. Reward: atom number (/ temperature if TOF is available)
        last_image = new_images[-1]
        raw_counts = np.sum(last_image * 255.0)
        atoms = atom_number(raw_counts, power=5, detuning=scaled_detuning, exposure=2)
        temperature = 1.0   # placeholder — replace with TOF result when available

        self.current_step += 1
        done = self.current_step >= self.episode_length
        reward = (atoms / temperature) / self.episode_length if done else 0.0

        info = {
            "atom_number": atoms,
            "temperature": temperature,
            "detuning": scaled_detuning,
            "step": self.current_step,
        }

        return self._get_observation(), reward, done, info

    # ------------------------------------------------------------------ #
    def _get_observation(self) -> Dict:
        stacked = np.stack(list(self.image_history), axis=0)   # (4, 50, 50)
        stacked = np.transpose(stacked, (1, 2, 0))             # (50, 50, 4)

        normalized_time = self.current_step / self.episode_length
        additional = np.array(
            [normalized_time, self._prev_detuning_normalized],
            dtype=np.float32,
        )
        return {"images": stacked, "additional": additional}