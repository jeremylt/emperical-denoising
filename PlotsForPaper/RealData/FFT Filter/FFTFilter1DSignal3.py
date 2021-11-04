# --------------------------------------------------
# Title: FFTFilter1D.py
# Date: 16 July 2014
# Author: Jeremy L Thompson
# Purpose: Implement FFT Soft (and Hard) Filter on
#          1D data
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

# Get threshold

thresh = 0.1

# Pad Data to power of 2
pad = 2
while pad < DataLen:
    pad *= 2

Data = np.append(Data, np.zeros(pad - DataLen))

OldDataLen = DataLen

DataLen = len(Data)

# ---- Slow DFT ----

def SFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(-2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast DFT ----

def FFT(x):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        return SFT(x)
    else:
        X_even = FFT(x[ : : 2])
        X_odd = FFT(x[1 : : 2])
        factor = np.exp(-2j*np.pi*np.arange(N)/N)

        return np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])


# ---- Slow iDFT ----

def iSFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast iDFT ----

def iFFT(x, initial):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        output = iSFT(x)
    else:
        X_even = iFFT(x[ : : 2], False)
        X_odd = iFFT(x[1 : : 2], False)
        factor = np.exp(2j*np.pi*np.arange(N)/N)

        output = np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])

    if initial == True:
        output /= N
        return output
    else:
        return output


# ---- FFT Data ----

FTData = FFT(Data)


# ---- Smooth Data ----

hardcut = thresh*np.std(FTData)

for i in range(DataLen):
    if np.abs(FTData[i]) < hardcut:
        FTData[i] = 0


# ---- Inverse FFT Data ----

SmoothedData = np.real(iFFT(FTData, True))

SmoothedData += datamean
Data += datamean


# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 3 \n FFT Filter - $\mu = 0.1 \sigma$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data[0 : OldDataLen])), Data[0 : OldDataLen], 'b', lw = 2)
plt.plot(range(len(SmoothedData[0 : OldDataLen])), SmoothedData[0 : OldDataLen], 'r', lw = 2)
plt.xlim([0, OldDataLen])

fig.set_size_inches(10, 4)
plt.savefig('FFTRealSignal3.png')

plt.show()
