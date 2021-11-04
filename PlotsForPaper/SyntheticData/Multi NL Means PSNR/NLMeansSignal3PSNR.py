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

    for i in range(1*DataLen/8, 2*DataLen/8):
        TrueData[i] += i - 1*DataLen/8
        Data[i] += i - 1*DataLen/8

    for i in range(3*DataLen/8, 4*DataLen/8):
        TrueData[i] += i - 3*DataLen/8
        Data[i] += i - 3*DataLen/8

    for i in range(6*DataLen/8, 7*DataLen/8):
        TrueData[i] += i - 6*DataLen/8
        Data[i] += i - 6*DataLen/8

    for i in range(DataLen):
        TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
        Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)

    return TrueData, Data


# ---- NL Means ----

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest

def NLMeansFilter(Data, beta, window, threshfactor):

    # ---- Set up NL Means Filter ----

    DataLen = len(Data)
    IntensityFilter = np.zeros(DataLen)

    offset = (window - 1)/2
    thresh = (max(Data) - min(Data))*(2*offset + 1)*threshfactor
    thresh *= thresh
    var = varest(Data)


    # ---- Filter the Data ----

    # Set Up Output
    SmoothedData = np.zeros(DataLen)

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter

        for j in range(i, DataLen - offset):
            norm = np.sum((Data[0 : i + offset + 1] - Data[j - i : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

        IntensityFilter = np.zeros(DataLen)

    # Middle
    IntensityFilter = np.zeros(DataLen)

    for i in range(offset, DataLen - offset):
        # Update Intensity Filter

        for j in range(offset, DataLen - offset):
            norm = np.sum((Data[i - offset : i + offset + 1] - Data[j - offset : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    # Right Hand Side
    IntensityFilter = np.zeros(DataLen)

    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter

        for j in range(offset, DataLen - (DataLen - i)):
            norm = np.sum((Data[i - offset : DataLen] - Data[j - offset : j + (DataLen - i)])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    return SmoothedData


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr

f = open("OutputSignal3.txt","a")

#f.write('Noise,Beta1,Window1,T1,Beta2,Window2,T2,PSNR\n')

for i in [1, 5, 10, 20, 30]:
    for j in [0.32, 0.34]:
        for k in [7]:
            for l in [0.65, 0.67]:
                for m in [0.32, 0.34]:
                    for n in [7]:
                        for o in [0.32, 0.34]:
                            for p in range(5):

                                TrueData, Data = buildsignal(i)
                                SmoothedDatai = NLMeansFilter(Data, j, k, l)
                                SmoothedData = NLMeansFilter(SmoothedDatai, m, n, o)

                                f.write(str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "," + str(m) + "," + str(n) + "," + str(o) + "," + str(psnr(SmoothedData, TrueData)) + "\n")

f.close()
