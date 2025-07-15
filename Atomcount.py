from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Function to calculate the number of atoms based on the count, power, detuning, and exposure
def atom_number(count, power, detuning, exposure):
    beam_radius = 0.4
    intensity = power / (np.pi * beam_radius * beam_radius)
    I_sat = 1.1
    s = 6 * intensity / I_sat
    gamma = 5.2
    aperture = 2.54
    scattering_rate = ((gamma / 2) * (10**6) * s) / (1 + s + 4 * (detuning / gamma)**2)
    solid_angle = (aperture / (4 * 10))**2
    quantum = 0.20
    atom_number_calc = count / (quantum * exposure * 0.001 * scattering_rate * solid_angle)
    return atom_number_calc

# Function to process images and calculate atom numbers
def process_image_and_get_N(image_path):
    k=1 # Counter for image naming
    df=[]
    Δ=15

    for dir in os.listdir(image_path):
        print(f"Processing directory: {dir}")

        folder_path = os.path.join(image_path, dir)
        
        if not os.path.isdir(folder_path): # Check if it's a directory
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            
            if not img_path.lower().endswith(('.bmp')): # Check if the file is a BMP image
                continue

            sname = "DT" + str(k) + " .jpg"
            k+=1
            
            image_array = Image.open(img_path)
            image_array = np.array(image_array)

            image_array = image_array[250:400,250:325] # Crop the image to the region of interest

            # Ensure no negative values after subtraction
            # image_array[image_array < 5] = 0

            # image_new = image_array_subtracted+image_new

            os.makedirs("images", exist_ok=True)
            plt.imshow(image_array, cmap='viridis')

            count = np.sum(image_array)
            N = atom_number(count,5,15,2)

            df.append({"atom_number": N,"detuning (MHz)": Δ,"image": img_path})
            
            plt.savefig("images/" + sname, bbox_inches='tight', pad_inches=0)
            plt.close()

    df=pd.DataFrame(df)
    df.to_csv("cnn_data.csv", index=False)
    print("Data saved to cnn_data.csv")
    return df

# df=process_image_and_get_N("data")