import numpy as np
import random as rnd

# -------- Generate Synthetic Signal --------

DataLen = 256
TrueData = np.zeros(DataLen)
Data = np.zeros(DataLen)

# ---- Set Parameters ----

mu = 25.0
amp1 = 10.0
amp2 = 2.0
per1 = 712.0
per2 = 128.0
jump = 12.0

noisesig = (amp1 + amp2 + jump)*10/100

# ---- Build Signal ----

for i in range(DataLen/4, 3*DataLen/4):
    TrueData[i] += jump
    Data[i] += jump

for i in range(DataLen):
    TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
    Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)


# -------- Wavelet Filter --------

# ---- DWT ----

def DWT(x):
    N = len(x)

    if N == 1 or N % 2 > 0:
        return x, N

    output = np.zeros(N)

    for i in range(N/2):
        output[i] = (x[i*2] + x[i*2 + 1])/2.
        output[N/2 + i] = (x[i*2] - x[i*2 + 1])/2.

    output[0 : N/2], length = DWT(output[0 : N/2])

    return output, length


# ---- iDWT ----

def iDWT(x, p, length):
    N = len(x)
    cap = 2**p*length

    if cap == N:
        return x

    output = np.zeros(N)
    output[2*cap : N] = x[2*cap : N]

    for i in range(cap):
        output[2*i] = x[i] + x[cap + i]
        output[2*i + 1] = x[i] - x[cap + i]

    output = iDWT(output, p + 1, length)

    return output


def WaveletFilter(Data, cutofffactor):

    # ---- DWT Data ----

    WTData, length = DWT(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(WTData)

    for i in range(DataLen):
        if np.abs(WTData[i]) < hardcut:
            WTData[i] = 0


    # ---- Inverse DWT Data ----

    WaveletFilterData = iDWT(WTData, 0, length)

    return WaveletFilterData

# ---- Results ----

dataavg = np.average(Data)
Data -= dataavg

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(3)
fig.suptitle('Simulated Data - 10% Noise',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})


# Plot small sigma
SmoothedData = WaveletFilter(Data, 0.5)

axarr[0].set_title('Wavelet Filter - $\mu$ = 1/2$\hat{\sigma}_n$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[0].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot medium sigma
SmoothedData = WaveletFilter(Data, 1.0)

axarr[1].set_title('Wavelet Filter - $\mu$ = $\hat{\sigma}_n$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[1].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot large sigma
SmoothedData = WaveletFilter(Data, 1.5)

axarr[2].set_title('Wavelet Filter - $\mu$ = 3/2$\hat{\sigma}_n$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[2].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('WaveletCompare.png')
plt.show()
