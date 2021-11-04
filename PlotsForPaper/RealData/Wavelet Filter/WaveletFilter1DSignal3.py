# --------------------------------------------------
# Title: WaveletFilter1D.py
# Date: 10 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement Wavelet Hard Thresholding
#          Filter on 1D data
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

datamean = np.average(Data)

Data -= datamean


# Get the threshold

thresh = 0.65


# ---- DWT ----

def DWT(x):
    N = len(x)

    if N == 1 or N % 2 > 0:
        return x, N

    output = np.zeros(N)

    for i in range(N/2):
        output[i] = (x[i*2] + x[i*2 + 1])/2.
        output[N/2 + i] = (x[i*2] - x[i*2 + 1])/2.

    output[0 : N/2], length = DWT(output[0 : N/2])

    return output, length


# ---- iDWT ----

def iDWT(x, p, length):
    N = len(x)
    cap = 2**p*length

    if cap == N:
        return x

    output = np.zeros(N)
    output[2*cap : N] = x[2*cap : N]

    for i in range(cap):
        output[2*i] = x[i] + x[cap + i]
        output[2*i + 1] = x[i] - x[cap + i]

    output = iDWT(output, p + 1, length)

    return output


# ---- DWT Data ----

WTData, length = DWT(Data)


# ---- Smooth Data ----

hardcut = thresh*np.std(WTData[length : DataLen])

for i in range(length, DataLen):
    if np.abs(WTData[i]) < hardcut:
        WTData[i] = 0


# ---- Inverse DWT Data ----

SmoothedData = iDWT(WTData, 0, length)

SmoothedData += datamean
Data += datamean


# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n Wavelet Filter - $\mu = 0.65 \sigma$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('WaveletRealSignal3.png')

plt.show()
