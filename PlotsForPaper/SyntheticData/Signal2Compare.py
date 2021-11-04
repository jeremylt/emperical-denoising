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


# ---- Results ----

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(5)
fig.suptitle('Simulated Data - Time Series 2',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})

DataLen = 256


# Plot 1%
axarr[0].set_title('1% Noise')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

TrueData, Data = buildsignal(1)

axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot 5%
axarr[1].set_title('5% Noise')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

TrueData, Data = buildsignal(5)

axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot 10%
axarr[2].set_title('10% Noise')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

TrueData, Data = buildsignal(10)

axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].set_xlim(0, DataLen)


# Plot 20%
axarr[3].set_title('20% Noise')
axarr[3].set_xlabel('Time Index')
axarr[3].set_ylabel('Signal')

TrueData, Data = buildsignal(20)

axarr[3].plot(range(DataLen), Data, 'b', lw = 2)
axarr[3].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[3].set_xlim(0, DataLen)


# Plot 30%
axarr[4].set_title('30% Noise')
axarr[4].set_xlabel('Time Index')
axarr[4].set_ylabel('Signal')

TrueData, Data = buildsignal(30)

axarr[4].plot(range(DataLen), Data, 'b', lw = 2)
axarr[4].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[4].set_xlim(0, DataLen)

fig.savefig('Signal2Compare.png')
plt.show()
