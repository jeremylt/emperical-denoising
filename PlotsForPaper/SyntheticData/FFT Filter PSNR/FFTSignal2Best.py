import numpy as np
import random as rnd

# -------- Generate Synthetic Signal --------

def buildsignal(pernoise):

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

    noisesig = (amp1 + amp2 + jump)*pernoise/100

    # ---- Build Signals ----

    for i in range(DataLen/3, DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(2*DataLen/3, 2*DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(DataLen):
        TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
        Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)

    return TrueData, Data


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
    DataLen = len(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(FTData)

    for i in range(DataLen):
        if np.abs(FTData[i]) < hardcut:
            FTData[i] = 0


    # ---- Inverse FFT Data ----

    FFTFilterData = np.real(iFFT(FTData, True))

    return FFTFilterData


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr


# ---- Plot Results ----

TrueData, Data = buildsignal(10)
SmoothedData = FFTFilter(Data, 0.15)
DataLen = len(Data)

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

psnrval = psnr(SmoothedData, TrueData)

fig.suptitle('Simulated Data Time Series 2 - 10%% Noise \n FFT Filter - $|\mu$ = $0.15 \sigma$ \n PSNR = %f' %psnrval,  fontsize = 12)
fig.set_size_inches(10, 4)
matplotlib.rcParams.update({'font.size': 10})

plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(DataLen), Data, 'b', lw = 2)
plt.plot(range(DataLen), TrueData, 'k', lw = 2)
plt.plot(range(DataLen), SmoothedData, 'r', lw = 2)
plt.plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
plt.xlim(0, DataLen)

fig.savefig('FFTSignal2Best.png')
plt.show()
