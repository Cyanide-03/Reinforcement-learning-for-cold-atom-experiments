from PIL import Image
import numpy as np
import os
import pandas as pd

# Function to calculate the number of atoms based on count, power, detuning, and exposure
def atom_number(count, power, detuning, exposure):
    beam_radius = 0.4
    intensity = power / (np.pi * beam_radius ** 2)
    I_sat = 1.1
    s = 6 * intensity / I_sat
    gamma = 5.2
    aperture = 2.54
    scattering_rate = ((gamma / 2) * 1e6 * s) / (1 + s + 4 * (detuning / gamma) ** 2)
    solid_angle = (aperture / (4 * 10)) ** 2
    quantum = 0.20
    atom_number_calc = count / (quantum * exposure * 0.001 * scattering_rate * solid_angle)
    return atom_number_calc

# Function to process images and calculate atom numbers
def process_image_and_get_N(image_path):
    k = 1
    data = []

    for dir in os.listdir(image_path):
        print(f"Processing directory: {dir}")
        folder_path = os.path.join(image_path, dir)
        if not os.path.isdir(folder_path):
            continue
        #get the folder name
        Δ = int(os.path.basename(folder_path))
        print(f"Detuing value: {Δ}") 

        for file in os.listdir(folder_path):

            img_path = os.path.join(folder_path, file)
            if not img_path.lower().endswith('.bmp'):
                continue
            print(f"Processing image: {img_path}")
            image = Image.open(img_path)  # Keep original mode (grayscale or RGB)
            image_array = np.array(image)

            # Crop the region of interest
            cropped = image_array[0:400, 90:400]

            # Calculate atom count
            count = np.sum(cropped)
            N = atom_number(count, power=5, detuning=15, exposure=2)

            # Prepare filename and save cropped image without changing colors
            sname = f"DT{k}.jpg"
            k += 1

            os.makedirs("Dataset/images", exist_ok=True)
            cropped_img = Image.fromarray(cropped)
            cropped_img.save(os.path.join("images", sname))

            data.append({
                "atom_number": N,
                "detuning (MHz)": Δ,
                "image": os.path.join("images", sname)
            })


    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    df.to_csv("Dataset/cnn_data.csv", index=False)
    print("Data saved to cnn_data.csv")
    return df

# # Run the function
# df = process_image_and_get_N("data")
# print(df.head()) # Display the first few rows of the DataFrame
