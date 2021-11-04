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


# -------- FFT Filter --------

# ---- Slow DFT ----

def SFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(-2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast DFT ----

def FFT(x):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        return SFT(x)
    else:
        X_even = FFT(x[ : : 2])
        X_odd = FFT(x[1 : : 2])
        factor = np.exp(-2j*np.pi*np.arange(N)/N)

        return np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])


# ---- Slow iDFT ----

def iSFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast iDFT ----

def iFFT(x, initial):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        output = iSFT(x)
    else:
        X_even = iFFT(x[ : : 2], False)
        X_odd = iFFT(x[1 : : 2], False)
        factor = np.exp(2j*np.pi*np.arange(N)/N)

        output = np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])

    if initial == True:
        output /= N
        return output
    else:
        return output

def FFTFilter(Data, cutofffactor):

    # ---- FFT Data ----

    FTData = FFT(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(FTData)

    for i in range(DataLen):
        if np.abs(FTData[i]) < hardcut:
            FTData[i] = 0


    # ---- Inverse FFT Data ----

    FFTFilterData = np.real(iFFT(FTData, True))

    return FFTFilterData

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
SmoothedData = FFTFilter(Data, 0.5)

axarr[0].set_title('FFT Filter - $\mu$ = 1/2$\hat{\sigma}_n$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[0].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot medium sigma
SmoothedData = FFTFilter(Data, 1.0)

axarr[1].set_title('FFT Filter - $\mu$ = $\hat{\sigma}_n$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[1].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot large sigma
SmoothedData = FFTFilter(Data, 1.5)

axarr[2].set_title('FFT Filter - $\mu$ = 3/2$\hat{\sigma}_n$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data + dataavg, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData + dataavg, 'r', lw = 2)
axarr[2].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('FFTCompare.png')
plt.show()
