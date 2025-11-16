import numpy as np
from scipy.interpolate import interp1d
import random
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import sys

from tensorflow.keras.initializers import GlorotUniform as KerasGlorotUniform, Zeros as KerasZeros
# Note: Keras 3 often exposes components directly under tf.keras or keras.initializers

class FixedGlorotUniform(KerasGlorotUniform):
    """
    Workaround: Fixes the serialization conflict by removing 'dtype'
    which newer Keras versions do not expect in the initializer's config.
    """
    def __init__(self, seed=None, **kwargs):
        # Crucially remove 'dtype' from the config before passing to the parent
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        super().__init__(seed=seed, **kwargs)

    # *** NEW CRITICAL STEP: Override the from_config method ***
    # This method is what Keras uses during deserialization.
    @classmethod
    def from_config(cls, config):
        # The config dict contains the arguments that failed earlier.
        # Use the class constructor to ensure 'dtype' is removed.
        return cls(**config)
class FixedZeros(KerasZeros):
    """
    Workaround: Fixes the serialization conflict for Zeros initializer.
    """
    def __init__(self, **kwargs):
        # Remove 'dtype' just like we did for GlorotUniform
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        # Zeros does not take 'seed', so we use a simple **kwargs pass
        # The parent Zeros() constructor takes no arguments
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        # Pass config to the constructor to ensure 'dtype' is removed
        return cls(**config)

class Simulation:
    """
    A simulation model for a Magneto-Optical Trap (MOT) experiment.

    This class uses pre-computed lookup tables (LUTs) from experimental data to
    simulate the atom number and temperature of the MOT as a function of laser detuning.
    It also uses a pre-trained Convolutional Neural Network (CNN) to generate
    synthetic fluorescence images of the atom cloud.
    """
    def __init__(self):
        # Use file-relative path so lookup files are found regardless of CWD
        BASE_DIR = os.path.dirname(__file__)
        exp_model_path = os.path.join(BASE_DIR, "lookup_table") + os.sep

        # --- Load Atom Number Data ---
        self.N_exp = np.squeeze(np.load(os.path.join(exp_model_path, 'LUT_N.npy')))
        self.det_N = np.squeeze(np.load(os.path.join(exp_model_path, 'det_N.npy')))
        self.N_max = np.squeeze(np.load(os.path.join(exp_model_path, 'N_max.npy')))

        # Normalize atom number for consistent scaling (0 to 1)
        self.N_exp = self.N_exp / self.N_max
        self.det_N = -self.det_N  # Detuning is typically negative
        
        # Create an interpolation function to predict atom number for any detuning
        self.N_intrp = interp1d(self.det_N, self.N_exp, 'cubic', bounds_error=False, fill_value=0)

        # --- Load Temperature Data ---
        self.T_exp = np.squeeze(np.load(os.path.join(exp_model_path, 'LUT_T.npy')))
        self.det_T = np.squeeze(np.load(os.path.join(exp_model_path, 'det_T.npy')))

        # Normalize temperature for consistent scaling
        self.T_exp = 0.1 * self.T_exp / self.T_exp[-1]
        self.det_T = -self.det_T  # Detuning is typically negative
        
        # Create an interpolation function for the logarithm of temperature for better stability
        self.logT_intrp = interp1d(self.det_T, np.log10(self.T_exp))

        self.det_max = max(self.det_N[-1], self.det_T[-1])

        model_filepath = os.path.join(BASE_DIR, "MOT_fluo_img_generator.h5")

        # --- Load Image Generation Model ---
        self.MOT_img_gen = None
        try:
            # Load the pre-trained Keras model for generating fluorescence images
            self.MOT_img_gen = load_model(model_filepath,
                                          # Pass the custom object to fix the deserialization error
                                          custom_objects={'GlorotUniform': FixedGlorotUniform,
                                                          'Zeros': FixedZeros
                                                        }
                                        )
            print("CNN model for image generation loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load CNN model for image generation. Detailed Error: {e}", file=sys.stderr)

    def predict_loading_rate(self, det):
        # ! multiplied by N max to return unnormalized atom number
        # Use the interpolation function and add some stochasticity
        return max(0.0, self.N_intrp(det) * random.gauss(1, 0.2)) # Ensure non-negative
        
    def predict_temperature(self, det):
        """Predicts the normalized temperature for a given detuning."""
        # # ! multiplied by T[-1]/0.1 to get the original unnormalized temp in K
        # Handle edge cases where detuning is outside the interpolation range
        if det >= max(self.det_T):
            return self.T_exp[0]
        elif det <= np.min(self.det_T):
            return self.T_exp[-1]
        else:
            # Use log-interpolation and convert back to linear scale
            return (10**self.logT_intrp(det))

    @tf.function(jit_compile=True)
    def fast_predict(self,x):
        return self.MOT_img_gen(x, training=False)


    def generate_image(self, norm_atom_number, norm_detuning):
        """
        Generates a synthetic fluorescence image using the pre-trained CNN.

        Args:
            norm_atom_number (float): Normalized atom number (0 to 1).
            norm_detuning (float): Normalized detuning value.

        Returns:
            np.ndarray: A 50x50 grayscale image as a float32 array.
        """

        if self.MOT_img_gen is None:
            # Return blank image if model not loaded
            return np.zeros((50, 50), dtype=np.float32)
        
        # Prepare input for the model (requires a batch dimension)
        input_data = tf.expand_dims(
            tf.stack([norm_atom_number, norm_detuning]), 
            axis=0
        )
        # img = self.MOT_img_gen.predict(input_data, verbose=0)
        img = self.fast_predict(input_data).numpy()
        
        # Squeeze and ensure proper shape
        img = img.squeeze()
        if len(img.shape) == 3:
            img = img[:, :, 0]
        
        # Post-process the generated image
        # Clip pixel values to the valid range [0.0, 1.0]
        img = np.clip(img, 0.0, 1.0)
        if norm_atom_number < 0.02:
            img = img * 0  # If atom number is negligible, the image should be black
            
        return img.astype(np.float32)
    