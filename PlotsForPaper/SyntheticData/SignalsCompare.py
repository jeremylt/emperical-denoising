import numpy as np
import random as rnd

# -------- Generate Synthetic Signal 1 --------

def buildsignal1(pernoise):

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


# -------- Generate Synthetic Signal 2 --------

def buildsignal2(pernoise):

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

    for i in range(DataLen/3, DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(2*DataLen/3, 2*DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(DataLen):
        TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
        Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)

    return TrueData, Data


# -------- Generate Synthetic Signal 3 --------

def buildsignal3(pernoise):

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


# ---- Calculate PSNR ----

def psnr(SmoothedData, TrueData):
    mse = np.average((SmoothedData - TrueData)**2)
    psnr = 20*np.log(np.max(SmoothedData)) - 10*np.log(mse)

    return psnr


# ---- Results ----

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(3)
fig.suptitle('Simulated Data',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})

DataLen = 256


# Plot Time Series 1
TrueData, Data = buildsignal1(10)

psnrval = psnr(Data, TrueData)

axarr[0].set_title('Time Series 1 - 10%% Noise - PSNR - %f' %psnrval)
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot Time Series 2
TrueData, Data = buildsignal2(10)

psnrval = psnr(Data, TrueData)

axarr[1].set_title('Time Series 2 - 10%% Noise - PSNR - %f' %psnrval)
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot Time Series 3
TrueData, Data = buildsignal3(10)

psnrval = psnr(Data, TrueData)

axarr[2].set_title('Time Series 3 - 10%% Noise - PSNR - %f' %psnrval)
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('SignalsCompare.png')
plt.show()
