import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.image import ssim

# Load dataset
data = pd.read_csv('dataset.csv')  
N_values = data['N'].values
delta_values = data['delta'].values
real_images = np.load('real_images.npy')  # Shape: (num_samples, 50, 50)

# Normalize inputs
N_max = np.max(N_values)
delta_max = 50  # Normalize δ assuming 50 MHz max
N_norm = N_values / N_max
delta_norm = delta_values / delta_max

# Stack inputs
X = np.stack((N_norm, delta_norm), axis=1)
y = real_images / 255.0  # Normalize pixel values
y = y.reshape((-1, 50, 50, 1))

# Load pre-trained model
model = load_model("MOT_fluo_img_generator.h5", compile=False)

# Define custom loss
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = tf.reduce_mean(1 - ssim(y_true, y_pred, max_val=1.0))
    return 0.5 * mse + 0.5 * ssim_loss

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss, metrics=[MeanSquaredError()])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.1)

# Save model
model.save("retrained_MOT_cesium_model.h5")

# Predict and visualize
predicted_image = model.predict(np.expand_dims(X[0], axis=0)).squeeze()
plt.imshow(predicted_image, cmap='gray')
plt.title("Generated Image")
plt.axis('off')
plt.show()
