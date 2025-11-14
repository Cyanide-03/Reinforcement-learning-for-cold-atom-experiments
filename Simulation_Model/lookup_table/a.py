import numpy as np

det_N = np.load('det_N.npy')
print("det_N.npy contents:")
print(det_N)

det_T = np.load('det_T.npy')
print("det_T.npy contents:")
print(det_T)

data = np.load('LUT_N.npy')
print("LUT_N.npy contents:")
print(data)

data = np.load('LUT_T.npy')
print("LUT_T.npy contents:")
print(data)

print(max(det_N[-1], det_T[-1]))