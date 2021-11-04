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

Data = DataSet[30, 5500 : 6500]

DataLen = len(Data)


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


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr


# ---- Plot Results ----

SmoothedData = BilateralFilter(Data, 2.0, 3.1)
SmoothedData2 = BilateralFilter(SmoothedData, 2.5, 2.4)

# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n First Bilateral Filter - $\sigma_d = 2.0$ - $\sigma_i = 3.1 \hat{\sigma}_n$ \n Second Bilateral Filter - $\sigma_d = 2.5$ - $\sigma_i = 2.4 \hat{\sigma}_n$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('2DiffBilateralRealSignal3.png')

plt.show()
