import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# check if model remembers optimizer, loss and trainable is true or not
model = load_model("MOT_fluo_img_generator.h5")
print(model.optimizer)
print(model.loss)
print(model.input_shape)
for layer in model.layers:
    print(layer.name, layer.trainable)


#Load dataset
from Atomcount import process_image_and_get_N
image_path = "to be set"
df = process_image_and_get_N(image_path)

#Preprocessing
print(df.head()) #check the 1st row If it contains column names then remove first row
X = df.iloc[ : , : -1 ].values
print(X.shape)
    # X.reshape() nedded if the X.shape is not same as model.input_shape
y = df.iloc[ : , -1 ].values
df['atom_number'] = df['atom_number'] / df['atom_number'].max()


# Training the model
epoch = 10 #to be decided
batch_size = 32 # to be decided
model.fit(X, y, epochs = epoch, batch_size = batch_size, validation_split = 0.2, verbose = 1)

# Save the model
model.save("MOT_fluo_img_generator_trained.h5")
