import numpy as np
from scipy.interpolate import interp1d
import random
from tensorflow.keras.models import load_model

class Simulation:
    def __init__(self):
        exp_model_path="Simulation_Model/lookup_table/"

        self.N_exp = np.squeeze(np.load(exp_model_path + 'LUT_N.npy'))
        self.det_N = np.squeeze(np.load(exp_model_path + 'det_N.npy'))
        self.N_max = np.squeeze(np.load(exp_model_path + 'N_max.npy'))

        self.N_exp = self.N_exp/self.N_max
        self.det_N = - self.det_N
        
        self.N_intrp = interp1d(self.det_N,self.N_exp,'cubic',bounds_error=False,fill_value=0)

        self.T_exp = np.squeeze(np.load(exp_model_path + 'LUT_T.npy'))
        self.det_T = np.squeeze(np.load(exp_model_path + 'det_T.npy'))

        self.T_exp = 0.1*self.T_exp/self.T_exp[-1] 
       
        self.det_T = - self.det_T
        
        self.logT_intrp = interp1d(self.det_T,np.log10(self.T_exp))


    def predict_loading_rate(self,det): # det is in [min,max]
        # Predicts the loading rate based on detuning
        return max((0,self.N_intrp(det)*random.gauss(1,0.2))) #
        
    def predict_temperature(self,det):
        # Predicts the temperature based on detuning
        if det>=max(self.det_T):
            return self.T_exp[0]
        elif det<=np.min(self.det_T):
            return self.T_exp[-1]
        else:
            return 10**self.logT_intrp(det)

    def generate_image(self,atom_number,detuning):
        model=load_model("Simulation_Model/MOT_fluo_img_generator.h5")
        detuning/=50
        atom_number/=self.N_max
        
        input_data=np.array([[atom_number,detuning]])
        img=model.predict(input_data)
        img=np.clip(img,0.0,1.0)

        return img[0,:,:,0]

