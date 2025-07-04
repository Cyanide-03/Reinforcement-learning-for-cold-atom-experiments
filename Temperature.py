# Temperture measurement...........................................


# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 13:26:11 2025

@author: IK
"""



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

import os

width =[]
count =[]

df = pd.DataFrame({'Width': width, 'Count': count})

excel_file_path = 'PGC_5m.xlsx'
df.to_excel(excel_file_path, index=False)

for i in range(1,51):

    number = i

    print(i)
    # Directory path where the image is located
    directory_path = r'C:\Users\poona\OneDrive\Desktop\python\mot charcterization\27_OCT'
    # Filename with a number
    filename = str(number)+" .bmp"
    sname = "DT"+str(number)+" .jpg"
    # Concatenate the path and filename
    image_path = os.path.join(directory_path, filename)

    # Open the image
    image_array = Image.open(image_path)

    image_array = np.array(image_array)

    image_array[image_array < 50] = 0

    image_array = image_array[150:650, 100:600]
    fit = np.sum(image_array, axis=0)
    x = np.arange(0,len(fit),1)
    def gauss_function(x, b,a, x0, sigma):
        return b+a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt,pcov = curve_fit(gauss_function, x, fit, [np.min(fit),np.max(fit)-np.min(fit),np.argmax(fit),20],maxfev = 1000000)
    plt.imshow(image_array,cmap ='viridis')
    plt.plot(x,-100*fit/np.max(fit))
    plt.savefig(sname)
    print(popt[3])
    plt.close()
    width.append(popt[3])

width = 0.0104* np.abs(width)
width = width*width
rl = int(len(width)/5)
revised =[]
std_revised =[]
for i in range(0,rl):
    a = 5*i
    b = 5*i +5
    revised.append(np.average(width[a:b]))
    std_revised.append(np.std(width[a:b]))


xx = np.arange(0,len(revised),1)
yy = revised
plt.errorbar(xx,yy,yerr=std_revised,color ='r',fmt ='o ')

def fit(x,a,b):
    return a*x*x+b

# Estimate initial guesses
initial_b = yy[0]  # Starting guess for b (intercept)
initial_a = (yy[-1] - yy[0]) / (xx[-1]**2 - xx[0]**2)  # Starting guess for a (curvature)

popt,pcov = curve_fit(fit, xx, yy, p0=[initial_a, initial_b],maxfev = 1000000)
x_00 = np.arange(0,len(revised),0.01)
plt.plot(x_00,fit(x_00,popt[0],popt[1]),color ='b')

print((popt[0]*133*1.6e-27)/(1.38e-23))

temp =(popt[0]*133*1.6e-27)/(1.38e-23)
er_temp =(np.sqrt(np.diag(pcov))[0]*133*1.6e-27)/(1.38e-23)
plt.title("temperature = "+ str(round((temp/1e-6),2))+" $ \pm  $"+str(round((er_temp/1e-6),2))+" $\mu K $",fontsize =12)
plt.xlabel('TOF (ms)',fontsize = 20)
plt.ylabel('  $ width^2 (mm^2)$', fontsize =20)
plt.tick_params(labelsize = '20')
plt.minorticks_on()
plt.tick_params(labelsize = 20,direction ='in',top= True,right =True,width = 1.5,which='both')
plt.rcParams["axes.linewidth"] = 1.5
# plt.legend(fontsize=12,loc='lower right')
plt.margins(x=0)
# plt.legend(fontsize =14)
plt.tight_layout()
