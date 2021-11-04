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

f = open("OutputSignal3.txt","a")

#f.write("Noise,SigD,SigI,PSNR\n")

for i in [1, 5, 10, 20, 30]:
    for j in [2.2, 2.3, 2.4]:
        for k in [2.8, 2.9, 3.0]:
            for l in range(5):

                TrueData, Data = buildsignal(i)
                SmoothedData = BilateralFilter(Data, j, k)
                SmoothedData2 = BilateralFilter(SmoothedData, j, k)

                f.write(str(i) + "," + str(j) + "," + str(k) + "," + str(psnr(SmoothedData2, TrueData)) + "\n")

f.close()
