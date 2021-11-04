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


# -------- Gaussian Filter --------

# Define Gaussian Function

def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))


def GaussianFilter(Data, sigma):

    # ---- Set up Gaussian Filter ----
    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigma) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    Filter = np.linspace(-offset, offset, filtersize)
    Filter = gaussian(Filter, sigma)
    Filter = Filter/np.sum(Filter) # Normalize


    # ---- Filter the Data ----

    # Set Up Output
    GaussianFilterData = np.array(np.zeros(DataLen))

    # Left Hand Side
    for i in range(0, offset):
        GaussianFilterData[i] = np.dot(Data[0 : i + offset + 1], Filter[offset - i : filtersize]/np.sum(Filter[offset - i : filtersize]))

    # Middle
    for i in range(offset, DataLen - offset):
        GaussianFilterData[i] = np.dot(Data[i - offset : i + offset + 1], Filter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        GaussianFilterData[i] = np.dot(Data[i - offset : DataLen], Filter[0 : filtersize - offset - i + DataLen - 1]/np.sum(Filter[0 : filtersize - offset - i + DataLen - 1]))

    return GaussianFilterData


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr


# ---- Plot Results ----

TrueData, Data = buildsignal(10)
SmoothedData = GaussianFilter(Data, 1.25)
DataLen = len(Data)

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

psnrval = psnr(SmoothedData, TrueData)

fig.suptitle('Simulated Data Time Series 3 - 10%% Noise \n Gaussian Filter - $\sigma_d$ = 1.25 \n PSNR = %f' %psnrval,  fontsize = 12)
fig.set_size_inches(10, 4)
matplotlib.rcParams.update({'font.size': 10})

plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(DataLen), Data, 'b', lw = 2)
plt.plot(range(DataLen), TrueData, 'k', lw = 2)
plt.plot(range(DataLen), SmoothedData, 'r', lw = 2)
plt.plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
plt.xlim(0, DataLen)

fig.savefig('GaussianSignal3Best.png')
plt.show()
