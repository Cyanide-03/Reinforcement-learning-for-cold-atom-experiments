import subprocess
import os
import time
import glob
import numpy as np
from PIL import Image
from typing import List, Optional

class ARTIQBridge:
    """
    Handles communication with ARTIQ hardware via command-line execution
    and image retrieval from a local directory.
    """
    def __init__(self, artiq_script: str = "mot_experiment.py", 
                 image_dir: str = "Dataset/real_time/",
                 poll_interval: float = 0.5):
        self.artiq_script = artiq_script
        self.image_dir = image_dir
        self.poll_interval = poll_interval
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def run_experiment(self, detuning: float) -> bool:
        """
        Executes the ARTIQ script with the specified detuning.
        Expects 'artiq_run' to be in the PATH.
        """
        try:
            # Example command: artiq_run mot_experiment.py -p detuning=-15.0
            command = ["artiq_run", self.artiq_script, "-p", f"detuning={detuning}"]
            print(f"Executing: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running ARTIQ script: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Error: 'artiq_run' command not found. Ensure ARTIQ is installed and in PATH.")
            return False

    def wait_for_new_images(self, count: int = 4, timeout: float = 30.0) -> List[np.ndarray]:
        """
        Polls the image directory for new images. 
        Returns a list of image arrays.
        """
        start_time = time.time()
        # Get existing files to distinguish new ones
        initial_files = set(glob.glob(os.path.join(self.image_dir, "*.bmp")))
        
        print(f"Waiting for {count} new images in {self.image_dir}...")
        
        while time.time() - start_time < timeout:
            current_files = set(glob.glob(os.path.join(self.image_dir, "*.bmp")))
            new_files = list(current_files - initial_files)
            
            if len(new_files) >= count:
                # Sort by modification time to get the actual most recent ones
                new_files.sort(key=os.path.getmtime)
                selected_files = new_files[-count:]
                
                images = []
                for f in selected_files:
                    try:
                        img = Image.open(f).convert('L')
                        img = img.resize((50, 50))
                        img_array = np.asarray(img, dtype=np.float32) / 255.0
                        images.append(img_array)
                    except Exception as e:
                        print(f"Error loading image {f}: {e}")
                
                if len(images) == count:
                    return images
            
            time.sleep(self.poll_interval)
            
        print(f"Timeout reached while waiting for images.")
        # Return blank images as fallback to prevent crash, but log the failure
        return [np.zeros((50, 50), dtype=np.float32) for _ in range(count)]

def get_latest_images_from_folder(folder: str, count: int = 4) -> List[np.ndarray]:
    """Simple utility to get newest images regardless of when they were created."""
    files = glob.glob(os.path.join(folder, "*.bmp"))
    if not files:
        files = glob.glob(os.path.join(folder, "*.jpg"))
    
    files.sort(key=os.path.getmtime, reverse=True)
    selected = files[:count]
    
    images = []
    for f in selected:
        img = Image.open(f).convert('L').resize((50, 50))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        
    while len(images) < count:
        images.append(np.zeros((50, 50), dtype=np.float32))
        
    return images
