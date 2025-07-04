# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 02:23:44 2024

@author: IK
"""


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

w = []
u = []
count_atom =[]
count = []
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

# Load the reference image


for i in range(2, 20): # Avg mean photon no
    number = i
    directory_path = r'C:\Users\poona\OneDrive\Desktop\python\mot charcterization\27_OCT'
    filename = str(number) + " .bmp"
    sname = "DT" + str(number) + " .jpg"
    image_path = os.path.join(directory_path, filename)

    # Open the current image
    image_array = Image.open(image_path)
    image_array = np.array(image_array) # Convert to 2D NumPy array


    image_array = image_array[0:400,0:600]
    # image_array = image_array[70:150,230:270]

    # Ensure no negative values after subtraction
    # image_array[image_array < 5] = 0

    # image_new = image_array_subtracted+image_new
    # Display the subtracted image
    plt.imshow(image_array, cmap='viridis')
    count = np.sum(image_array)
    abc = atom_number(count,5,15,2)
    u.append(count)
    count_atom.append(abc)
    # Save the subtracted image as a file
    plt.savefig(sname)
    plt.close()


count_mean = np.average(count_atom)
count_std = np.std(count_atom)
photon_count = np.average(u)
photon_count_std = np.std(u)
print("atom mean:", count_mean)
print("atom std:",count_std)
print("photon counts:" ,photon_count)
print("photon counts error:" ,photon_count_std)
plt.show()
# print(u)

# print(count_atom)



