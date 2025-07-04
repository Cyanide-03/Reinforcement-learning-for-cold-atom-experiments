import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("MOT_fluo_img_generator.h5", compile=False)

# Dummy test input — make sure to adapt this to match training input shape/range
predicted_image=model.predict(np.expand_dims((49000000,224.760219),axis=0))*255

# Generate image using model
predicted_image = np.clip(predicted_image, 0, 255).astype(np.uint8)
predicted_image = np.squeeze(predicted_image)  # Shape becomes (50, 50)

# Apply Gaussian blur to smooth the image (adjust kernel size as needed)
smoothed_image = cv2.GaussianBlur(predicted_image, (5, 5), 0)

# Convert grayscale image to RGB
colored_image = cv2.applyColorMap(predicted_image, cv2.COLORMAP_JET)
rgb_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

# Plot and save image
# Plot and save the RGB image
plt.imshow(rgb_image)
plt.title("Generated Fluorescence Image")
plt.axis('off')
plt.savefig("generated_fluo_image_rgb.png", bbox_inches='tight')
print("Image saved as generated_fluo_image_rgb.png")
