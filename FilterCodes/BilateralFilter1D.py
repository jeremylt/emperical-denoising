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
f = open("2010_10_gen.txt","r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append(numbers)

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

# Extract the 4th (data) row
Data = []
Data = np.array([int(i) for i in transArr[3][1:]])

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

print 'The estimated noise standard deviation is ' + str(np.sqrt(varest(Data)))

sigmadist = float(raw_input("Please enter the desired spatial kernel standard deviation: "))

sigmaint = float(raw_input("Please enter the desired intensity kernel standard deviation: "))

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
import matplotlib.pyplot as plt

plt.title('Actual Data vs Smoothed Data',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Wind Power Generated')

plt.plot(range(DataLen), Data, 'k', lw = 3)
plt.plot(range(len(SmoothedData)), SmoothedData, 'b')

plt.show()

Noise = SmoothedData - Data

plt.title('Calculated Noise',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Calculated Noise')

plt.plot(range(DataLen), Noise, 'b')

print 'Noise Mean: ' + str(np.average(Noise))
print 'Noise Std Dev: ' + str(np.std(Noise))

plt.show()


# ---- Loop for Updating Smoothed Data ----
'''
 Note: this would be run when a new data point is received (rather than re-running the whole function for all data)

# Update Intensity Filter
IntensityFilter = Data[i - offset : DataLen] - Data[i]
IntensityFilter = gaussian(IntensityFilter, sigmaint)
IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen + offset) - 1])
IntensityFilter *= 1/sum(IntensityFilter) # Normalize

# Smooth Next Data Point
SmoothedData.append(np.dot(Data[i - offset : DataLen], IntensityFilter)

for i in range(DataLen - offset - 1, DataLen - 1):
    # Update Intensity Filter
    IntensityFilter = Data[i - offset : DataLen] - Data[i]
    IntensityFilter = gaussian(IntensityFilter, sigmaint)
    IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen - 1 + offset) - 1])
    IntensityFilter *= 1/sum(IntensityFilter) # Normalize

    # Smooth Next Data Point
    SmoothedData[i] = np.dot(Data[i - offset : DataLen], IntensityFilter)
'''
