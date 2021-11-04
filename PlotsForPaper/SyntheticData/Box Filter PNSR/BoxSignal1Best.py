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

    for i in range(DataLen/4, 3*DataLen/4):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(DataLen):
        TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
        Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)

    return TrueData, Data


# -------- Box Filter --------

def BoxFilter(Data, Window):
    # ---- Set Up Moving Average

    window = 5
    halfwindow = (window - 1)/2

    DataLen = len(Data)

    BoxFilter = np.ones(window)/window

    # ---- Filter the Data ----

    BoxFilterData = np.zeros(DataLen)

    # Left Hand Side
    for i in range(0, halfwindow):
        BoxFilterData[i] = np.sum(Data[0 : i + halfwindow + 1])/(i + halfwindow + 1)

    # Middle
    for i in  range(halfwindow, DataLen - halfwindow):
        BoxFilterData[i] = np.dot(Data[i - halfwindow : i + halfwindow + 1], BoxFilter)

    # Right Hand Side
    for i in range(DataLen - halfwindow, DataLen):
        BoxFilterData[i] = np.sum(Data[i - halfwindow : DataLen])/(DataLen - i + halfwindow)

    return BoxFilterData


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr


# ---- Plot Results ----

TrueData, Data = buildsignal(10)
SmoothedData = BoxFilter(Data, 5)
DataLen = len(Data)

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

psnrval = psnr(SmoothedData, TrueData)

fig.suptitle('Simulated Data Time Series 1 - 10%% Noise \n Box Filter - $|I|$ = 5 \n PSNR = %f' %psnrval,  fontsize = 12)
fig.set_size_inches(10, 4)
matplotlib.rcParams.update({'font.size': 10})

plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(DataLen), Data, 'b', lw = 2)
plt.plot(range(DataLen), TrueData, 'k', lw = 2)
plt.plot(range(DataLen), SmoothedData, 'r', lw = 2)
plt.plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
plt.xlim(0, DataLen)

fig.savefig('BoxSignal1Best.png')
plt.show()
