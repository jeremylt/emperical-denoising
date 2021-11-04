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
f = open("2010_10_gen.txt","r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append(numbers)

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

# Extract the 4th (data) row
Data = []
DataTemp = []
DataTemp = np.array([int(i) for i in transArr[3][1:]])

Data = DataTemp[6800 : 7300]

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

beta = float(raw_input("Please enter the desired smoothing level (beta): "))

window = int(raw_input("Please enter the desired window size (odd number): "))

offset = (window - 1)/2

print "The maximum difference in intensiy is " + str(max(Data) - min(Data))

thresh = float(raw_input("Please enter the desired preselection threshold: "))


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
# Note: this would be run when a new data point is received (rather than re-running the whole function for all data)

for i in range(DataLen - offset - 1, DataLen):
    print float(i)/DataLen
    # Update Intensity Filter

    for j in range(offset, DataLen - (DataLen - i)):
        if np.norm(Data[i - offset : DataLen] - Data[j - offset : j + (DataLen - i)]) < thresh:
            norm = np.linalg.norm(Data[i - offset : DataLen] - Data[j - offset : j + (DataLen - i)])**2
            IntensityFilter[j] = np.exp(-norm/(2*beta*sigma**2*(2*offset + 1)))
        else:
            IntensityFilter[j] = 0

    IntensityFilter[i] = 1

    IntensityFilter *= 1/sum(IntensityFilter)

    # Smooth Next Data Point
    SmoothedData[i] = np.dot(Data, IntensityFilter)
'''
