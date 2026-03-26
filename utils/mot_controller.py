"""
mot_controller.py  —  ARTIQ RPC Server (runs on the ARTIQ / hardware machine)
==============================================================================
Start this ONCE before training begins:

    python mot_controller.py

It stays alive the entire training session. The RL agent connects to it
and calls set_detuning_and_capture() remotely via sipyco RPCServer.

Requirements:
    pip install sipyco
    (sipyco is bundled with ARTIQ; if you have ARTIQ installed it's already there)
"""

import os
import time
import glob
import subprocess
import numpy as np
from PIL import Image
from sipyco.pc_rpc import simple_server_loop   # sipyco is part of ARTIQ


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — adjust these to match your setup
# ─────────────────────────────────────────────────────────────────────────────
ARTIQ_SCRIPT   = "mot_exp.py"   # The ARTIQ experiment script
IMAGE_DIR      = "Dataset/real_time/"  # Folder where the camera drops BMP files
IMAGE_SIZE     = 50                    # Resize target for fluorescence images
NUM_IMAGES     = 4                     # How many images to capture per step
POLL_INTERVAL  = 0.2                   # seconds between folder polls
IMAGE_TIMEOUT  = 30.0                  # seconds to wait before giving up
SERVER_HOST    = "0.0.0.0"            # listen on all interfaces
SERVER_PORT    = 3386                  # port the RL agent will connect to
# ─────────────────────────────────────────────────────────────────────────────


class MOTController:
    """
    This class is exposed over RPC.
    Every public method here becomes callable from the RL agent (the client).
    """

    def __init__(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print("[MOTController] Ready. Waiting for RPC calls...")

    # ------------------------------------------------------------------ #
    #  Primary method called by the RL agent at every environment step    #
    # ------------------------------------------------------------------ #
    def set_detuning_and_capture(self, detuning: float) -> dict:
        """
        1. Send detuning value to ARTIQ hardware (runs the experiment script).
        2. Wait for the camera to write new BMP images to IMAGE_DIR.
        3. Load, resize, and return the images as a flat list + metadata.

        Returns a dict (plain Python types only — RPC serialises to JSON):
            {
                "success":  bool,
                "images":   list[list[list[float]]],   # shape [4][50][50]
                "detuning": float
            }
        """
        print(f"[MOTController] set_detuning_and_capture({detuning:.3f} Γ)")

        # ── Step 1: record which files already exist so we can detect new ones ──
        existing = set(glob.glob(os.path.join(IMAGE_DIR, "*.bmp")))

        # ── Step 2: trigger ARTIQ experiment ─────────────────────────────────
        success = self._run_artiq(detuning)

        # ── Step 3: wait for new images ──────────────────────────────────────
        # images_np = self._wait_for_new_images(existing, count=NUM_IMAGES)

        # ── Step 4: convert numpy arrays → plain Python lists for RPC ────────
        # images_list = [img.tolist() for img in images_np]

        return {
            "success":  success,
            # "images":   images_list,   # [4][50][50] nested list of floats
            "detuning": detuning,
        }

    def ping(self) -> str:
        """Simple health-check the client can call to verify connection."""
        return "pong"

    def get_latest_images(self) -> dict:
        """
        Return the 4 most recently modified BMP files without triggering
        a new ARTIQ run.  Useful for the initial environment reset().
        """
        files = sorted(
            glob.glob(os.path.join(IMAGE_DIR, "*.bmp")),
            key=os.path.getmtime,
            reverse=True
        )[:NUM_IMAGES]

        images_np = []
        for f in files:
            img = Image.open(f).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            images_np.append((np.asarray(img, dtype=np.float32) / 255.0).tolist())

        # Pad with blank images if not enough files exist yet
        while len(images_np) < NUM_IMAGES:
            images_np.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32).tolist())

        return {"images": images_np}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _run_artiq(self, detuning: float) -> bool:
        """Call artiq_run as a subprocess (synchronous — blocks until done)."""
        cmd = ["artiq_run", 
               ARTIQ_SCRIPT]
        print(f"  → Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] artiq_run failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print("  [ERROR] 'artiq_run' not found in PATH.")
            return False

    def _wait_for_new_images(self, existing_files: set, count: int) -> list:
        """
        Poll IMAGE_DIR until `count` new BMP files appear (compared to
        existing_files), then load and return them as numpy arrays.
        Falls back to blank images on timeout.
        """
        deadline = time.time() + IMAGE_TIMEOUT

        while time.time() < deadline:
            current = set(glob.glob(os.path.join(IMAGE_DIR, "*.bmp")))
            new_files = sorted(current - existing_files, key=os.path.getmtime)

            if len(new_files) >= count:
                selected = new_files[-count:]
                images = []
                for f in selected:
                    try:
                        img = Image.open(f).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
                        images.append(np.asarray(img, dtype=np.float32) / 255.0)
                    except Exception as e:
                        print(f"  [WARN] Could not load {f}: {e}")
                if len(images) == count:
                    print(f"  → Got {count} new images.")
                    return images

            time.sleep(POLL_INTERVAL)

        print(f"  [WARN] Timeout waiting for images. Returning blank frames.")
        return [np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) for _ in range(count)]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = MOTController()

    print(f"\n[MOTController] Starting RPC server on {SERVER_HOST}:{SERVER_PORT}")
    print("[MOTController] Press Ctrl-C to stop.\n")

    # simple_server_loop blocks forever and serves RPC requests
    # "MOTController" is the name clients use to look up this object
    simple_server_loop(
        {"MOTController": controller},
        SERVER_HOST,
        SERVER_PORT,
    )