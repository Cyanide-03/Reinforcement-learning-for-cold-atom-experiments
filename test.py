import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("MOT_fluo_img_generator_trained2.h5")

# Load and preprocess csv data
df = pd.read_csv("cnn_data.csv")  # Previously saved from process_image_and_get_N()

# Normalize features (must match training)
N_max = df['atom_number'].max()
Δ_max = df['detuning (MHz)'].max()
df['atom_number'] = df['atom_number'] / N_max
df['detuning (MHz)'] = df['detuning (MHz)'] / 50

X = df.iloc[:, :2].values  # (atom_number, detuning)

# Predict using the model
predicted_images = model.predict(X)
print("Predicted images shape:", predicted_images.shape)

# Visualize first few predictions
# plt.imshow(predicted_images.squeeze(), cmap='gray')
# plt.title(f"Generated Image")
# plt.axis('off')
# plt.savefig(f"gen2.png", bbox_inches='tight', pad_inches=0)
# plt.close()

predicted_images = predicted_images[0].squeeze()
predicted_images = (predicted_images * 255).astype(np.uint8)
img = Image.fromarray(predicted_images, 'L' )
img_resized = img.resize((720, 1440), Image.BILINEAR)
img_resized.save("gen2_resized.png")
