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


# ---- NL Means ----

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest

def NLMeansFilter(Data, beta):

    # ---- Set up NL Means Filter ----

    DataLen = len(Data)
    IntensityFilter = np.zeros(DataLen)

    offset = 3
    thresh = (max(Data) - min(Data))*(2*offset + 1)*0.5
    thresh *= thresh
    var = varest(Data)


    # ---- Filter the Data ----

    # Set Up Output
    SmoothedData = np.zeros(DataLen)

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter

        for j in range(i, DataLen - offset):
            norm = np.sum((Data[0 : i + offset + 1] - Data[j - i : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

        IntensityFilter = np.zeros(DataLen)

    # Middle
    IntensityFilter = np.zeros(DataLen)

    for i in range(offset, DataLen - offset):
        # Update Intensity Filter

        for j in range(offset, DataLen - offset):
            norm = np.sum((Data[i - offset : i + offset + 1] - Data[j - offset : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    # Right Hand Side
    IntensityFilter = np.zeros(DataLen)

    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter

        for j in range(offset, DataLen - (DataLen - i)):
            norm = np.sum((Data[i - offset : DataLen] - Data[j - offset : j + (DataLen - i)])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    return SmoothedData


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
SmoothedData = NLMeansFilter(Data1, 0.5)

axarr[0].set_title('Non-Local Means Filter - $\\beta$ = 0.5, $|I|$ = 7, $T$ = $0.75 ( max Y_j - min Y_j)$')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen1), Data1, 'b', lw = 2)
axarr[0].plot(range(DataLen1), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen1)


# Plot Time Series 2
SmoothedData = NLMeansFilter(Data2, 0.5)

axarr[1].set_title('Non-Local Means Filter - $\\beta$ = 0.5, $|I|$ = 7, $T$ = $0.75 ( max Y_j - min Y_j)$')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen2), Data2, 'b', lw = 2)
axarr[1].plot(range(DataLen2), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen2)


# Plot Time Series 3
SmoothedData = NLMeansFilter(Data3, 0.5)

axarr[2].set_title('Non-Local Means Filter - $\\beta$ = 0.5, $|I|$ = 7, $T$ = $0.75 ( max Y_j - min Y_j)$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen3), Data3, 'b', lw = 2)
axarr[2].plot(range(DataLen3), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen3)

fig.savefig('NLMeansRealCompare.png')
plt.show()
