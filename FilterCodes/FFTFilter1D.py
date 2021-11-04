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
f = open("2010_10_gen.txt","r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append(numbers)

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

# Extract the 4th (data) row
Data = []
Data = [int(i) for i in transArr[3][1:]]

DataLen = len(Data)

datamean = np.average(Data)

Data -= datamean

# Get threshold

thresh = float(raw_input("Please enter the desired FFT threshold: "))

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

import matplotlib.pyplot as plt

plt.title('Actual Data vs Smoothed Data',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Wind Power Generated')

plt.plot(range(OldDataLen), Data[0 : OldDataLen], 'k', lw = 3)
plt.plot(range(OldDataLen), SmoothedData[0 : OldDataLen], 'b')

plt.show()

Noise = SmoothedData - Data

plt.title('Calculated Noise',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Calculated Noise')

plt.plot(range(OldDataLen), Noise[0 : OldDataLen], 'b')

print 'Noise Mean: ' + str(np.average(Noise))
print 'Noise Std Dev: ' + str(np.std(Noise))

plt.show()
