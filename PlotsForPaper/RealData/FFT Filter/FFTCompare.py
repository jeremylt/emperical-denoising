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


Data1 = DataSet[54, 4600 : 5500]

DataLen1 = len(Data1)


Data2 = DataSet[12, 7800 : ]

DataLen2 = len(Data2)


Data3 = DataSet[30, 5300 : 6500]

DataLen3 = len(Data3)


# -------- FFT Filter --------

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

def FFTFilter(Data, cutofffactor):

    # ---- FFT Data ----

    FTData = FFT(Data)

    DataLen = len(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(FTData)

    for i in range(DataLen):
        if np.abs(FTData[i]) < hardcut:
            FTData[i] = 0


    # ---- Inverse FFT Data ----

    FFTFilterData = np.real(iFFT(FTData, True))

    return FFTFilterData


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
SmoothedData = FFTFilter(Data1, 1)

axarr[0].set_title('FFT Filter - $\\alpha$ = 1')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen1), Data1, 'b', lw = 2)
axarr[0].plot(range(DataLen1), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen1)


# Plot Time Series 2
SmoothedData = FFTFilter(Data2, 1)

axarr[1].set_title('FFT Filter - $\\alpha$ = 1')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen2), Data2, 'b', lw = 2)
axarr[1].plot(range(DataLen2), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen2)


# Plot Time Series 3
SmoothedData = FFTFilter(Data3, 1)

axarr[2].set_title('FFT Filter - $\\alpha$ = 1')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen3), Data3, 'b', lw = 2)
axarr[2].plot(range(DataLen3), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen3)

fig.savefig('FFTRealCompare.png')
plt.show()
