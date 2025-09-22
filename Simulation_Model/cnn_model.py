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

# check if model remembers optimizer, loss and trainable is true or not
BASE_DIR = os.path.dirname(__file__)  # folder of cnn_model.p
model_path = os.path.join(BASE_DIR, "MOT_fluo_img_generator.h5")

model = load_model(model_path)
# print(model.summary())
# print(model.optimizer)
# print(model.loss)
# print(model.input_shape)
# for layer in model.layers:
#     print(layer.name, layer.trainable)


#Load dataset
from utilitis.Atomcount import process_image_and_get_N
image_path = ["Dataset/data"]

for i in range(len(image_path)):
    # df = process_image_and_get_N(image_path[i])
    df=pd.read_csv("cnn_data.csv")

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42)

    #Preprocessing
    print(df.head()) 

    N_max= df['atom_number'].max()
    df['atom_number'] = df['atom_number'] / N_max
    Δ_max = df['detuning (MHz)'].max()
    df['detuning (MHz)'] = df['detuning (MHz)'] / 50

    X = df.iloc[ : , : -1 ].values
    print("Input Shape: ",X.shape) # (406, 2) - 2 features: power and detuning


    def load_image(image_path):
        img=Image.open(image_path).convert('L')  # Convert to grayscale
        # print("Image shape: ", img.size)  # Should be (50, 50)
        img=img.resize((50, 50))  # Resize to match model output shape
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = img_array[..., np.newaxis] 
        return img_array

    y = np.stack([load_image(p) for p in df['image']], axis=0)
    print("Output shape: ",y.shape)  # (406, 50, 50, 1) - 50x50 grayscale images

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

    # Visualize some images
    # plt.imshow(y[0].squeeze(), cmap='gray')
    # plt.title(f"Actual Image")
    # plt.axis('off')
    # plt.savefig(f"k.png", bbox_inches='tight', pad_inches=0)
    # plt.close()

    # Training the model
    epoch = 100 #to be decided
    batch_size = 32 # to be decided
    callbacks = [
        ModelCheckpoint("MOT_best.h5", save_best_only=True, monitor='val_loss', verbose=1, save_format="h5"),
        EarlyStopping(patience=10, min_delta=1e-3, monitor='val_loss', restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    history = model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, callbacks=callbacks, validation_split = 0.2, verbose = 2)

    # Save the model
    print(f"Model trained and saved for {image_path[i]}")

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss vs Epoch for {image_path[i]}")
    plt.legend()
    plt.grid(True)
    plot_filename = f"loss_plot_{image_path[i]}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved as {plot_filename}")

    # -------------- Evaluation on test set --------------
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE:", mse)

    # Predict images
    y_pred = model.predict(X_test, verbose=2)
    y_pred=np.clip(y_pred,0.0,1.0)

    # Compute SSIM & PSNR for a few samples
    ssim_scores, psnr_scores = [], []
    for j in range(len(y_test)):
        true_img = y_test[j].squeeze()
        pred_img = y_pred[j].squeeze()
        
        ssim_score = ssim(true_img, pred_img, data_range=1.0)  # images normalized [0,1]
        psnr_score = psnr(true_img, pred_img, data_range=1.0)
        
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
        axes[idx, 0].imshow(y_test[idx].squeeze(), cmap='gray')
        axes[idx, 0].set_title('GT'); axes[idx, 0].axis('off')
        axes[idx, 1].imshow(y_pred[idx].squeeze(), cmap='gray')
        axes[idx, 1].set_title('Pred'); axes[idx, 1].axis('off')
    plt.tight_layout()
    plt.savefig(f"sample_pred_{image_path[i]}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved sample_pred_{image_path[i]}.png")