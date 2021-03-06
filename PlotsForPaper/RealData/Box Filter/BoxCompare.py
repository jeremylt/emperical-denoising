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


Data1 = DataSet[54, 4800 : 5500]

DataLen1 = len(Data1)


Data2 = DataSet[12, 8000 : ]

DataLen2 = len(Data2)


Data3 = DataSet[30, 5500 : 6500]

DataLen3 = len(Data3)


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
fig.suptitle('Real Data',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})


# Plot Time Series 1
SmoothedData = BoxFilter(Data1, 7)

axarr[0].set_title('Box Filter - $|I|$ = 7')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen1), Data1, 'b', lw = 2)
axarr[0].plot(range(DataLen1), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen1)


# Plot Time Series 2
SmoothedData = BoxFilter(Data2, 7)

axarr[1].set_title('Box Filter - $|I|$ = 7')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen2), Data2, 'b', lw = 2)
axarr[1].plot(range(DataLen2), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen2)


# Plot Time Series 3
SmoothedData = BoxFilter(Data3, 7)

axarr[2].set_title('Box Filter - $|I|$ = 7')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen3), Data3, 'b', lw = 2)
axarr[2].plot(range(DataLen3), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen3)

fig.savefig('BoxRealCompare.png')
plt.show()
