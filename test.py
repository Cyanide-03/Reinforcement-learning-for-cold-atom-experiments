import pandas as pd;
import numpy as np;

df = pd.read_csv("cnn_data.csv")
N = 562740.8785611945/df['atom_number'].max()
delta = 15/50;

from tensorflow.keras.models import load_model
model = load_model("MOT_fluo_img_generator_trained.h5")
img = model.predict(np.expand_dims((N,delta),axis=0))*255


