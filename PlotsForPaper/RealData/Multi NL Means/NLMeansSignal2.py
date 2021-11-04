import numpy as np

# ---- Data Input ----

# Read data into a matrix
arr = []
f = open("2011_10.dat","r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append([float(val) for val in numbers])

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

DataSet = np.array(transArr)

Data = DataSet[12, 8000 : ]

DataLen = len(Data)


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


# ---- Plot Results ----

SmoothedDatai = NLMeansFilter(Data, 0.23, 7, 0.74)
SmoothedData = NLMeansFilter(SmoothedDatai, 0.35, 7, 0.51)

# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 2 \n First Non-Local Means Filter - $\\beta = 0.23$ - $|I| = 7$ - $T = 0.74 ( max Y_j - min Y_j )$ \n Second Non-Local Means Filter - $\\beta = 0.35$ - $|I| = 7$ - $T = 0.51 ( max Y_j - min Y_j )$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('MultiNLMeansRealSignal2.png')

plt.show()
