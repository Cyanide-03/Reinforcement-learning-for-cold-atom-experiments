import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# This script trains a Convolutional Neural Network (CNN) to generate synthetic
# fluorescence images of a Magneto-Optical Trap (MOT). The model takes normalized
# atom number and laser detuning as input and outputs a 50x50 grayscale image.

# --- Load Pre-trained Model ---
BASE_DIR = os.path.dirname(__file__)  # folder of cnn_model.p
model_path = os.path.join(BASE_DIR, "MOT_fluo_img_generator.h5")

model = load_model(model_path)
# print(model.summary())
# print(model.optimizer)
# print(model.loss)
# print(model.input_shape)
# for layer in model.layers:
#     print(layer.name, layer.trainable)


# --- Load and Preprocess Dataset ---
from utils.Atomcount import process_image_and_get_N, atom_number
image_path = ["Dataset/data"]

for i in range(len(image_path)):
    # df = process_image_and_get_N(image_path[i])
    df=pd.read_csv("Dataset/cnn_data.csv")

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42)

    # --- Preprocessing ---
    print(df.head()) 

    # Normalize the input features (atom number and detuning) to a [0, 1] range
    # This is crucial for stable neural network training.
    N_max= df['atom_number'].max()
    df['atom_number'] = df['atom_number'] / N_max
    Δ_max = df['detuning (MHz)'].max()
    df['detuning (MHz)'] = df['detuning (MHz)'] / 50

    # X contains the input features for the model
    X = df.iloc[ : , : -1 ].values
    print("Input Shape: ",X.shape) # (406, 2) - 2 features: power and detuning


    def load_image(image_path):
        """Helper function to load, resize, and normalize an image."""
        img=Image.open(image_path).convert('L')  # Convert to grayscale
        img=img.resize((50, 50))  # Resize to match model output shape
        img_array = np.asarray(img, dtype=np.float32) / 255.0 # Normalize pixel values to [0, 1]
        img_array = img_array[..., np.newaxis] # Add a channel dimension for the CNN
        return img_array

    # y contains the target images
    y = np.stack([load_image(os.path.join("Dataset/",p)) for p in df['image']], axis=0)
    print("Output shape: ",y.shape)  # (406, 50, 50, 1) - 50x50 grayscale images

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);


    # --- Model Training ---
    epoch = 100 #to be decided
    batch_size = 32 # to be decided
    callbacks = [
        ModelCheckpoint("MOT_best.h5", save_best_only=True, monitor='val_loss', verbose=1), # Save the best model based on validation loss
        EarlyStopping(patience=10, min_delta=1e-3, monitor='val_loss', restore_best_weights=True, verbose=1), # Stop training if validation loss doesn't improve
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1) # Reduce learning rate if training plateaus
    ]
    history = model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, callbacks=callbacks, validation_split = 0.2, verbose = 2)

    # Save the model
    print(f"Model trained and saved for {image_path[i]}")

    plt.figure(figsize=(8, 6))
    # Plot training and validation loss to check for overfitting
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss vs Epoch for {image_path[i]}")
    plt.legend()
    plt.grid(True)
    plot_filename = f"loss_plot.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved as {plot_filename}")

    # --- Evaluation on the Test Set ---
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE:", mse)

    # Predict images
    y_pred = model.predict(X_test, verbose=2)
    y_pred=np.clip(y_pred,0.0,1.0) # Ensure predicted pixel values are in the valid [0, 1] range

    # Compute SSIM & PSNR for a few samples
    ssim_scores, psnr_scores = [], []
    for j in range(len(y_test)):
        true_img = y_test[j].squeeze()
        pred_img = y_pred[j].squeeze()
        
        ssim_score = ssim(true_img, pred_img, data_range=1.0)  # Structural Similarity Index
        psnr_score = psnr(true_img, pred_img, data_range=1.0)  # Peak Signal-to-Noise Ratio
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    print("Average SSIM:", np.mean(ssim_scores))
    print("Average PSNR:", np.mean(psnr_scores))

    # Test MSE → how close predicted pixels are
    # SSIM (0–1) → how structurally similar generated images are to ground truth
    # PSNR (higher = better, usually 20–40 dB is decent)

    # save a small gallery of examples (ground truth vs predicted)
    n_show = min(6, len(y_test))
    fig, axes = plt.subplots(n_show, 2, figsize=(6, 3*n_show))
    for idx in range(n_show):
        # Get ground truth image
        true_img = y_test[idx].squeeze()*255
        true_img = np.clip(true_img, 0, 255).astype(np.uint8)
        
        # Un-normalize the atom number for display
        true_atom_number = X_test[idx][0] * N_max 

        # Predicted image
        pred_img = y_pred[idx].squeeze()*255
        pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)

        # Calculate atom number for predicted image
        orig_height, orig_width = 400, 310 # Original dimensions before cropping
        pred_height, pred_width = pred_img.shape
        scale_factor = (orig_height * orig_width) / (pred_height * pred_width)
        count = np.sum(pred_img) * scale_factor
        detuning = X_test[idx][1] * 50  # un-normalize detuning
        pred_atom_number = atom_number(count, power=5, detuning=detuning, exposure=2)

        # Plot ground truth
        axes[idx, 0].imshow(true_img, cmap='gray')
        axes[idx, 0].set_title(f'GT\nAtoms: {true_atom_number:.0f}')
        axes[idx, 0].axis('off')

        # Plot predicted image
        axes[idx, 1].imshow(pred_img, cmap='gray')
        axes[idx, 1].set_title(f'Pred\nAtoms: {pred_atom_number:.0f}')
        axes[idx, 1].axis('off')
    plt.tight_layout()
    plt.savefig(f"sample_pred.png", bbox_inches='tight')
    plt.close()
    print(f"Saved sample_pred.png")