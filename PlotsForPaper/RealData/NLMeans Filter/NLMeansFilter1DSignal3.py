# --------------------------------------------------
# Title: NLMeansFilter1D.py
# Date: 24 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement NL Means Filter on 1D data
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


# ---- Set up NL Means Filter ----

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest

var = varest(Data)

IntensityFilter = np.zeros(DataLen)

# Get the parameters

beta = 0.5

window = 9

offset = (window - 1)/2

thresh = 0.75


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


# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n Non-Local Means Filter - $\\beta = 0.5$ - $|I| = 9$ - $T = 0.75 ( max Y_j - min Y_j )$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('NLMeansRealSignal3.png')

plt.show()
