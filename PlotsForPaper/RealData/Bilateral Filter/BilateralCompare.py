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


Data1 = DataSet[54, 4800 : 5500]

DataLen1 = len(Data1)


Data2 = DataSet[12, 8000 : ]

DataLen2 = len(Data2)


Data3 = DataSet[30, 5500 : 6500]

DataLen3 = len(Data3)


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
fig.suptitle('Real Data',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})


# Plot Time Series 1
SmoothedData = BilateralFilter(Data1, 2.5, 1)

axarr[0].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 2.5 $\hat{\sigma}_n$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen1), Data1, 'b', lw = 2)
axarr[0].plot(range(DataLen1), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen1)


# Plot Time Series 2
SmoothedData = BilateralFilter(Data2, 2.5, 2)

axarr[1].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 2.5 $\hat{\sigma}_n$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen2), Data2, 'b', lw = 2)
axarr[1].plot(range(DataLen2), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen2)


# Plot Time Series 3
SmoothedData = BilateralFilter(Data3, 2.5, 3)

axarr[2].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 2.5 $\hat{\sigma}_n$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen3), Data3, 'b', lw = 2)
axarr[2].plot(range(DataLen3), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen3)

fig.savefig('BilateralRealCompare.png')
plt.show()
