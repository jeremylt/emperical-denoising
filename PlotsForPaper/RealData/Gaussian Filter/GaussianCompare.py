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
SmoothedData = GaussianFilter(Data1, 2.25)

axarr[0].set_title('Gaussian Filter - $\sigma_d$ = 2.25')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen1), Data1, 'b', lw = 2)
axarr[0].plot(range(DataLen1), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen1)


# Plot Time Series 2
SmoothedData = GaussianFilter(Data2, 2.25)

axarr[1].set_title('Gaussian Filter - $\sigma_d$ = 2.25')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen2), Data2, 'b', lw = 2)
axarr[1].plot(range(DataLen2), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen2)


# Plot Time Series 3
SmoothedData = GaussianFilter(Data3, 2.25)

axarr[2].set_title('Gaussian Filter - $\sigma_d$ = 2.25')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen3), Data3, 'b', lw = 2)
axarr[2].plot(range(DataLen3), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen3)

fig.savefig('GaussianRealCompare.png')
plt.show()
