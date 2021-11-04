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

    DataLen = len(Data)
    WTData, length = DWT(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(WTData)

    for i in range(DataLen):
        if np.abs(WTData[i]) < hardcut:
            WTData[i] = 0


    # ---- Inverse DWT Data ----

    WaveletFilterData = iDWT(WTData, 0, length)

    return WaveletFilterData


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr

f = open("OutputSignal2.txt","a")

#f.write("Noise,Thresh,PSNR\n")

for i in [1, 5, 10, 20, 30]:
    for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for k in range(10):

            TrueData, Data = buildsignal(i)
            SmoothedData = WaveletFilter(Data, j)

            f.write(str(i) + "," + str(j) + "," + str(psnr(SmoothedData, TrueData)) + "\n")

f.close()
