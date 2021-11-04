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


# -------- Bilateral Filter --------

# Define Gaussian Function

def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))

# Define Variance Estimator

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest


def BilateralFilter(Data, sigmadist, sigintfac):

    # ---- Set Up Gaussian Filter ----
    sigmaint = sigintfac*np.sqrt(varest(Data))

    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigmadist) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    SpatialFilter = np.linspace( - offset, offset, filtersize)
    SpatialFilter = gaussian(SpatialFilter, sigmadist)
    SpatialFilter *= 1/np.sum(SpatialFilter) # Normalize


    # ---- Filter the Data ----

    # Set Up Output
    BilateralFilterData = np.zeros(DataLen)

    IntensityFilter = np.array([])

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter
        IntensityFilter = Data[0 : i + offset + 1] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[offset - i : filtersize])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[0 : i + offset + 1], IntensityFilter)

    # Middle
    for i in range(offset, DataLen - offset):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : i + offset + 1] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter)
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : i + offset + 1], IntensityFilter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : DataLen] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen + offset) - 1])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : DataLen], IntensityFilter)

    return BilateralFilterData


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
SmoothedData = BilateralFilter(Data, 2.5, 1)

axarr[0].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = $\hat{\sigma}_n$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[0].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot medium sigma
SmoothedData = BilateralFilter(Data, 2.5, 2)

axarr[1].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 2$\hat{\sigma}_n$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[1].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot large sigma
SmoothedData = BilateralFilter(Data, 2.5, 3)

axarr[2].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 3$\hat{\sigma}_n$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[2].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('BilateralCompare.png')
plt.show()
