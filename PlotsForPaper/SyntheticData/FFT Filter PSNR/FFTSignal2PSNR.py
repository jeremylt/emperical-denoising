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

f = open("OutputSignal2.txt","a")

#f.write("Noise,Thresh,PSNR\n")

for i in [1, 5, 10, 20, 30]:
    for j in [0.1, 0.15, 0.2, 0.75, 0.85, 0.95, 1.0]:
        for k in range(5):

            TrueData, Data = buildsignal(i)
            SmoothedData = FFTFilter(Data, j)

            f.write(str(i) + "," + str(j) + "," + str(psnr(SmoothedData, TrueData)) + "\n")

f.close()
