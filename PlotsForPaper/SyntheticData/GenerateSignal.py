import numpy as np
import random as rnd

# -------- Generate Synthetic Signal --------

DataLen = 512
TrueData = np.zeros(DataLen)
Data = np.zeros(DataLen)

# ---- Set Parameters ----

noisesig = 1

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


# ---- Results ----

# Plotting results
import matplotlib.pyplot as plt

plt.title('Synthetic Signal',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(DataLen), Data, 'b')
plt.plot(range(DataLen), TrueData, 'k', lw = 2)
plt.xlim(0, DataLen)


plt.show()
