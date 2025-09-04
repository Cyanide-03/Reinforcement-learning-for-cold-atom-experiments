import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# check if model remembers optimizer, loss and trainable is true or not
model = load_model("MOT_fluo_img_generator.h5")
# print(model.summary())
# print(model.optimizer)
# print(model.loss)
# print(model.input_shape)
# for layer in model.layers:
#     print(layer.name, layer.trainable)


#Load dataset
from Atomcount import process_image_and_get_N
image_path = ["data"]

for i in range(len(image_path)):
    df = process_image_and_get_N(image_path[i])

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
    history = model.fit(X_train, y_train, epochs = epoch, batch_size = batch_size, validation_split = 0.2, verbose = 1)

    # Save the model
    model.save("MOT_fluo_img_generator_trained" + image_path[i] + ".h5")
    print(f"Model trained and saved for dataset {image_path[i]}")

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