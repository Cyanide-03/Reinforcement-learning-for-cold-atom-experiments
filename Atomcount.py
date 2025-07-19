from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
    Δ = 15

    for dir in os.listdir(image_path):
        print(f"Processing directory: {dir}")
        folder_path = os.path.join(image_path, dir)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            if not img_path.lower().endswith('.bmp'):
                continue

            image = Image.open(img_path)
            image_array = np.array(image)

            # Crop the region of interest
            cropped = image_array[250:400, 250:325]

            # Calculate atom count
            count = np.sum(cropped)
            N = atom_number(count, power=5, detuning=15, exposure=2)

            # Prepare filename and save cropped image without axes
            sname = f"DT{k}.jpg"
            k += 1

            os.makedirs("images", exist_ok=True)
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(cropped, cmap='viridis')

            save_path = os.path.join("images", sname)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            data.append({
                "atom_number": N,
                "detuning (MHz)": Δ,
                "image": save_path
            })

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    df.to_csv("cnn_data.csv", index=False)
    print("Data saved to cnn_data.csv")
    return df

# df = process_image_and_get_N("data")