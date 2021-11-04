# --------------------------------------------------
# Title: BilateralFilter1D.py
# Date: 24 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement Bilateral Filter on 1D data
# --------------------------------------------------


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


# ---- Set up Gaussian Filter ----

def varest(Data):
    DataLen = len(Data)

    varest = 0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2

    varest *= 2./(3.*float(DataLen - 2))

    return varest

# Get the sigmas

sigmadist = 2.75

sigmaint = 2.75 * np.sqrt(varest(Data))

# Define Gaussian Function
def gaussian(x, sigma):
    return 1/((2*np.pi)**0.5*sigma)*np.exp(-x**2./(2*sigma**2.))

# Build Gaussian Filter
filtersize = 2*int(5*sigmadist) + 1
filtersize = min(filtersize, DataLen - 1) # prevent overrun

offset = (filtersize - 1)/2

SpatialFilter = np.linspace(-offset, offset, filtersize)
SpatialFilter = gaussian(SpatialFilter, sigmadist)
SpatialFilter *= 1/np.sum(SpatialFilter) # Normalize


# ---- Filter the Data ----

# Set Up Output
SmoothedData = np.zeros(DataLen)

IntensityFilter = np.array([])

# Left Hand Side
for i in range(0, offset):
    # Update Intensity Filter
    IntensityFilter = Data[0 : i + offset + 1] - Data[i]
    IntensityFilter = gaussian(IntensityFilter, sigmaint)
    IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[offset - i : filtersize])
    IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

    # Smooth Next Data Point
    SmoothedData[i] = np.dot(Data[0 : i + offset + 1], IntensityFilter)

# Middle
for i in range(offset, DataLen - offset):
    # Update Intensity Filter
    IntensityFilter = Data[i - offset : i + offset + 1] - Data[i]
    IntensityFilter = gaussian(IntensityFilter, sigmaint)
    IntensityFilter = np.multiply(IntensityFilter, SpatialFilter)
    IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

    # Smooth Next Data Point
    SmoothedData[i] = np.dot(Data[i - offset : i + offset + 1], IntensityFilter)

# Right Hand Side
for i in range(DataLen - offset, DataLen):
    # Update Intensity Filter
    IntensityFilter = Data[i - offset : DataLen] - Data[i]
    IntensityFilter = gaussian(IntensityFilter, sigmaint)
    IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen + offset) - 1])
    IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

    # Smooth Next Data Point
    SmoothedData[i] = np.dot(Data[i - offset : DataLen], IntensityFilter)


# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n Bilateral Filter - $\sigma_d = 2.75$ - $\sigma_i = 2.75 \hat{\sigma}_n$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('BilateralRealSignal3.png')

plt.show()
