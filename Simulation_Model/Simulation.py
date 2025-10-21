import numpy as np
from scipy.interpolate import interp1d
import random
from tensorflow.keras.models import load_model
import os

class Simulation:
    def __init__(self):
        # Use file-relative path so lookup files are found regardless of CWD
        BASE_DIR = os.path.dirname(__file__)
        exp_model_path = os.path.join(BASE_DIR, "lookup_table") + os.sep

        self.N_exp = np.squeeze(np.load(os.path.join(exp_model_path, 'LUT_N.npy')))
        self.det_N = np.squeeze(np.load(os.path.join(exp_model_path, 'det_N.npy')))
        self.N_max = np.squeeze(np.load(os.path.join(exp_model_path, 'N_max.npy')))

        self.N_exp = self.N_exp / self.N_max
        self.det_N = -self.det_N
        
        self.N_intrp = interp1d(self.det_N, self.N_exp, 'cubic', bounds_error=False, fill_value=0)

        self.T_exp = np.squeeze(np.load(os.path.join(exp_model_path, 'LUT_T.npy')))
        self.det_T = np.squeeze(np.load(os.path.join(exp_model_path, 'det_T.npy')))

        self.T_exp = 0.1 * self.T_exp / self.T_exp[-1]
        self.det_T = -self.det_T 
        
        self.logT_intrp = interp1d(self.det_T, np.log10(self.T_exp))

        self.MOT_img_gen = None
        try:
            self.MOT_img_gen = load_model(os.path.join(BASE_DIR, "MOT_fluo_img_generator.h5"))
        except:
            print("Warning: Could not load CNN model for image generation")

    def predict_loading_rate(self, det):

        return max(0, self.N_intrp(det) * random.gauss(1, 0.2))
        
    def predict_temperature(self, det):

        if det >= max(self.det_T):
            return self.T_exp[0]
        elif det <= np.min(self.det_T):
            return self.T_exp[-1]
        else:
            return 10**self.logT_intrp(det)

    def generate_image(self, atom_number, detuning):

        if self.MOT_img_gen is None:
            # Return blank image if model not loaded
            return np.zeros((50, 50), dtype=np.float32)
        
        normalized_atoms = atom_number / 25
        normalized_detuning = -detuning / 50
        
        input_data = np.expand_dims([normalized_atoms, normalized_detuning], axis=0)
        img = self.MOT_img_gen.predict(input_data, verbose=0)
        
        # Squeeze and ensure proper shape
        img = img.squeeze()
        if len(img.shape) == 3:
            img = img[:, :, 0]
        
        # Clip to valid range and handle very low atom numbers
        img = np.clip(img, 0.0, 1.0)
        if atom_number < 0.02:
            img = img * 0  # Zero out image for very low atom counts
            
        return img.astype(np.float32)