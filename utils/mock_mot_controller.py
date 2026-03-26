"""
mock_mot_controller.py  —  Fake ARTIQ Server for Testing
=========================================================
Simulates the MOTController RPC server WITHOUT any real hardware.

Run this instead of mot_controller.py during testing:
    python mock_mot_controller.py

It will:
  - Accept RPC connections on the same port (3386)
  - Simulate a realistic atom number vs detuning curve (Lorentzian peak)
  - Generate fake fluorescence images (Gaussian blob that grows/shrinks with atom number)
  - Add realistic noise so the RL agent actually has something to learn
  - Print every call so you can watch the loop in real-time

Once this works end-to-end, swap in the real mot_controller.py — zero other changes needed.
"""

import os
import time
import math
import random
import numpy as np
from sipyco.pc_rpc import simple_server_loop

# ── Server config (must match artiq_bridge.py) ────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3386

# ── Simulated physics parameters ──────────────────────────────────────────
OPTIMAL_DETUNING   = 20.0   # Γ  — peak atom number here
PEAK_ATOM_NUMBER   = 5e7    # atoms at optimal detuning
LINEWIDTH          = 8.0    # Γ  — FWHM of the Lorentzian response curve
NOISE_FRACTION     = 0.05   # 5% shot-to-shot noise
IMAGE_SIZE         = 50
NUM_IMAGES         = 4
SIMULATED_DELAY    = 0.3    # seconds — mimics hardware latency per step


class MockMOTController:
    """
    Fake hardware controller with the exact same public API as MOTController.
    The RL agent (via ARTIQBridge) cannot tell the difference.
    """

    def __init__(self):
        self._call_count = 0
        print("[MockMOTController] Ready — simulating MOT hardware.")
        print(f"  Optimal detuning: {OPTIMAL_DETUNING} Γ")
        print(f"  Peak atom number: {PEAK_ATOM_NUMBER:.1e}")
        print(f"  Simulated step delay: {SIMULATED_DELAY}s\n")

    # ── Primary RPC method (matches real MOTController) ───────────────────
    def set_detuning_and_capture(self, detuning: float) -> dict:
        self._call_count += 1
        time.sleep(SIMULATED_DELAY)  # mimic hardware latency

        atoms = self._simulate_atom_number(detuning)
        images_np = self._generate_fake_images(atoms, count=NUM_IMAGES)
        images_list = [img.tolist() for img in images_np]

        print(f"  [Mock #{self._call_count:04d}] detuning={detuning:6.2f} Γ  "
              f"→  atoms={atoms:.3e}")

        return {
            "success": True,
            "images":  images_list,
            "detuning": detuning,
        }

    def ping(self) -> str:
        return "pong"

    def get_latest_images(self) -> dict:
        """Called during env.reset() — return images at a mid-range detuning."""
        atoms = self._simulate_atom_number(OPTIMAL_DETUNING * 0.7)
        images_np = self._generate_fake_images(atoms, count=NUM_IMAGES)
        return {"images": [img.tolist() for img in images_np]}

    # ── Physics simulation ────────────────────────────────────────────────
    def _simulate_atom_number(self, detuning: float) -> float:
        """
        Lorentzian peak centred at OPTIMAL_DETUNING.
        Matches the shape of a real MOT loading curve vs detuning.
        """
        delta = detuning - OPTIMAL_DETUNING
        lorentzian = 1.0 / (1.0 + (2.0 * delta / LINEWIDTH) ** 2)
        noise = 1.0 + random.gauss(0, NOISE_FRACTION)
        return max(0.0, PEAK_ATOM_NUMBER * lorentzian * noise)

    def _generate_fake_images(self, atom_number: float,
                               count: int = 4) -> list:
        """
        Generate grayscale fluorescence images:
          - Gaussian blob centred in the frame
          - Peak intensity scales with atom_number
          - Small frame-to-frame jitter and Poisson-like noise
        """
        images = []
        peak_intensity = min(1.0, atom_number / PEAK_ATOM_NUMBER)

        for _ in range(count):
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

            # Blob centre jitter (±2 pixels)
            cx = IMAGE_SIZE / 2 + random.gauss(0, 1.5)
            cy = IMAGE_SIZE / 2 + random.gauss(0, 1.5)
            sigma = 8.0 + random.gauss(0, 0.5)   # blob width in pixels

            y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            gaussian = (gaussian * peak_intensity).astype(np.float32)

            # Add Poisson-like shot noise
            shot_noise = np.random.normal(0, 0.01, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
            img = np.clip(gaussian + shot_noise, 0.0, 1.0)
            images.append(img)

        return images


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = MockMOTController()

    print(f"[MockMOTController] RPC server starting on port {SERVER_PORT}")
    print("[MockMOTController] Press Ctrl-C to stop.\n")

    simple_server_loop(
        {"MOTController": controller},   # same target_name as real server
        SERVER_HOST,
        SERVER_PORT,
    )