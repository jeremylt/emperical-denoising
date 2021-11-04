import numpy as np
import random as rnd

# -------- Generate Synthetic Signal --------

DataLen = 512
TrueData = np.zeros(DataLen)
Data = np.zeros(DataLen)

# ---- Set Parameters ----

noisesig = 2

mu = 35
amp1 = 10
amp2 = 2
per1 = 1048
per2 = 256

# ---- Build Signal ----

for i in range(DataLen):
    TrueData[i] = mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)

for i in range(DataLen):
    Data[i] = mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)


# ---- Calculate PSNR ----

mse = np.average((Data - TrueData)**2)

psnr = 20*np.log(np.max(Data)) - 10*np.log(mse)

print psnr
