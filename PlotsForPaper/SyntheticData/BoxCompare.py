import numpy as np
import random as rnd

# -------- Generate Synthetic Signal --------

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

pernoise = 10

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


# ---- Results ----

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(3)
fig.suptitle('Simulated Data - 10% Noise',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})


# Plot small sigma
SmoothedData = BoxFilter(Data, 3)

axarr[0].set_title('Box Filter - $|I|$ = 3')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[0].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot medium sigma
SmoothedData = BoxFilter(Data, 5)

axarr[1].set_title('Box Filter - $|I|$ = 5')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[1].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot large sigma
SmoothedData = BoxFilter(Data, 7)

axarr[2].set_title('Box Filter - $|I|$ = 7')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[2].plot(range(DataLen), Data - SmoothedData, 'c', lw = 2)
axarr[2].set_xlim(0, DataLen)

fig.savefig('BoxCompare.png')
plt.show()
