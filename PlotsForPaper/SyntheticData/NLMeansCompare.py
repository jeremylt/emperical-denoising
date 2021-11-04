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

pernoise = 10

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


# ---- NL Means ----

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest

def NLMeansFilter(Data, beta):

    # ---- Set up NL Means Filter ----

    DataLen = len(Data)
    IntensityFilter = np.zeros(DataLen)

    offset = 3
    thresh = (max(Data) - min(Data))*(2*offset + 1)*0.5
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


# ---- Results ----

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
SmoothedData = NLMeansFilter(Data, 0.5)

axarr[0].set_title(r'NLMeans Filter - $\beta$ = 0.5, $|I$| = 7, $T$ = 0.5 $( max Y_j - min Y_j )$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[0].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot medium sigma
SmoothedData = NLMeansFilter(Data, 0.75)

axarr[1].set_title(r'NLMeans Filter - $\beta$ = 0.75, $|I|$ = 7, $T$ = 0.5 $( max Y_j - min Y_j )$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[1].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot large sigma
SmoothedData = NLMeansFilter(Data, 1.0)

axarr[2].set_title(r'NLMeans Filter - $\beta$ = 1.0, $|I|$ = 7, $T$ = 0.5 $( max Y_j - min Y_j )$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[2].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('NLMeansCompare.png')
plt.show()
