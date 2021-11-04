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

SmoothedDatai = BilateralFilter(Data, 2.5, 1.6)
SmoothedData = NLMeansFilter(SmoothedDatai, 0.5, 7, 0.9)

# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n Bilateral Filter - $\sigma_d = 2.5$ - $\sigma_i = 1.6 \hat{\sigma}_n$ \n Non-Local Means Filter - $\\beta = 0.5$ - $|I| = 7$ - $T = 0.8 ( max Y_j min Y_j )$',  fontsize = 12)
fig.set_size_inches(10, 4)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

plt.savefig('BilateralNLMeansRealSignal3.png')

plt.show()
