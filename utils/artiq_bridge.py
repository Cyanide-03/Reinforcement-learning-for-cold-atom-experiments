import subprocess
import os
import time
import glob
import numpy as np
from PIL import Image
from typing import List, Optional

"""
artiq_bridge.py  —  RPC Client (runs on the RL / training machine)
===================================================================
Replaces the old subprocess + folder-polling approach with a direct
sipyco RPC connection to the MOTController server.
 
The server (mot_controller.py) must be running before you start training.
 
Usage (same API as before — RealMOTenv.py does NOT need to change its calls):
 
    bridge = ARTIQBridge(host="localhost", port=3386)
    result = bridge.run_experiment_and_get_images(detuning=25.0)
    images = result["images"]   # list of 4 numpy arrays, shape (50,50)
"""
 
try:
    from sipyco.pc_rpc import Client as RPCClient
    SIPYCO_AVAILABLE = True
except ImportError:
    SIPYCO_AVAILABLE = False
    print("[ARTIQBridge] WARNING: sipyco not installed. "
          "Install with: pip install sipyco   (or it comes with ARTIQ)")
 
 
IMAGE_SIZE = 50
 
 
class ARTIQBridge:
    """
    RPC client that talks to the MOTController server running on the
    ARTIQ / hardware machine.
 
    Drop-in replacement for the old ARTIQBridge — the public interface
    (run_experiment, wait_for_new_images) is kept so RealMOTenv.py works
    unchanged, but internally we use RPC instead of subprocess.
    """
 
    def __init__(self,
                 host: str = "localhost",
                 port: int = 3386,
                 # Legacy args kept for compatibility — no longer used
                 artiq_script: str = "mot_experiment.py",
                 image_dir: str = "Dataset/real_time/"):
 
        self.host = host
        self.port = port
        self._client: Optional[RPCClient] = None
 
        # Cache for images returned by the last RPC call
        # so that run_experiment() and wait_for_new_images() can stay separate
        self._last_images: List[np.ndarray] = []
        self._last_success: bool = False
 
        if not SIPYCO_AVAILABLE:
            raise RuntimeError(
                "sipyco is required for RPC communication. "
                "Install it with: pip install sipyco"
            )
 
        self._connect()
 
    # ------------------------------------------------------------------ #
    #  Connection management                                               #
    # ------------------------------------------------------------------ #
    def _connect(self):
        """Open the RPC connection to MOTController."""
        print(f"[ARTIQBridge] Connecting to MOTController at {self.host}:{self.port} ...")
        try:
            self._client = RPCClient(self.host, self.port, target_name="MOTController")
            reply = self._client.ping()
            print(f"[ARTIQBridge] Connected! Server replied: {reply}")
        except Exception as e:
            raise ConnectionError(
                f"[ARTIQBridge] Could not connect to MOTController at "
                f"{self.host}:{self.port}.\n"
                f"  Make sure mot_controller.py is running on the hardware machine.\n"
                f"  Original error: {e}"
            )
 
    def close(self):
        """Close the RPC connection gracefully."""
        if self._client is not None:
            self._client.close_rpc()
            self._client = None
            print("[ARTIQBridge] RPC connection closed.")
 
    def __del__(self):
        self.close()

    # def __init__(self, artiq_script: str = "mot_experiment.py", 
    #              image_dir: str = "Dataset/real_time/",
    #              poll_interval: float = 0.5):
    #     self.artiq_script = artiq_script
    #     self.image_dir = image_dir
    #     self.poll_interval = poll_interval
        
    #     if not os.path.exists(self.image_dir):
    #         os.makedirs(self.image_dir)

    # ------------------------------------------------------------------ #
    #  Public API  (matches what RealMOTenv.py calls)                     #
    # ------------------------------------------------------------------ #
    def run_experiment(self, detuning: float) -> bool:
        """
        Trigger the ARTIQ experiment with the given detuning.
        Also fetches the resulting images and caches them internally
        so that wait_for_new_images() can return them immediately.
 
        Returns True on success, False on hardware error.
        """
        try:
            result = self._client.set_detuning_and_capture(float(detuning))
        except Exception as e:
            print(f"[ARTIQBridge] RPC call failed: {e}")
            self._last_images = [np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                                  for _ in range(4)]
            self._last_success = False
            return False
 
        # Convert nested lists → numpy arrays
        # self._last_images = [
        #     np.array(img, dtype=np.float32) for img in result["images"]
        # ]
        self._last_success = result["success"]
        return self._last_success

    def wait_for_new_images(self, count: int = 4, timeout: float = 30.0) -> List[np.ndarray]:
        """
        Returns the images captured during the last run_experiment() call.
 
        With RPC, images are already available by the time run_experiment()
        returns, so this method just hands back the cached result.
        The `timeout` parameter is kept for API compatibility but is unused.
        """
        if not self._last_images:
            print("[ARTIQBridge] WARNING: wait_for_new_images() called before "
                  "run_experiment(). Returning blank images.")
            return [np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                    for _ in range(count)]
 
        return self._last_images[:count]

    def get_initial_images(self) -> List[np.ndarray]:
            """
            Fetch the latest images from the hardware folder without triggering
            a new experiment.  Used during environment reset() to get a real
            starting state instead of blank frames.
            """
            try:
                result = self._client.get_latest_images()
                return [np.array(img, dtype=np.float32) for img in result["images"]]
            except Exception as e:
                print(f"[ARTIQBridge] get_initial_images RPC failed: {e}")
                return [np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) for _ in range(4)]

# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test — run this file directly to check the connection
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 3386
 
    print(f"\n=== ARTIQBridge connection test ===")
    bridge = ARTIQBridge(host=host, port=port)
 
    print("\nTesting run_experiment(detuning=20.0) ...")
    ok = bridge.run_experiment(20.0)
    print(f"  Success: {ok}")
 
    imgs = bridge.wait_for_new_images(count=4)
    print(f"  Got {len(imgs)} images, shapes: {[i.shape for i in imgs]}")
    print(f"  Pixel value range: {imgs[0].min():.3f} – {imgs[0].max():.3f}")
 
    bridge.close()
    print("\nTest complete.")
 