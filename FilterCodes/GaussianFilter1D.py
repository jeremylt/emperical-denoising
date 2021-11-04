# --------------------------------------------------
# Title: GaussianFilter1D.py
# Date: 24 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement Gaussian Filter on 1D data
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


# ---- Set up Gaussian Filter ----

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest

# Get the sigma

print 'The estimated noise standard deviation is ' + str(varest(Data)**(1/2))

sigma = float(raw_input("Please enter the desired spatial kernel standard deviation: "))

# Define Gaussian Function
def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))

# Build Gaussian Filter
filtersize = 2*int(5*sigma) + 1
filtersize = min(filtersize, DataLen - 1) # prevent overrun

offset = (filtersize - 1)/2

Filter = np.linspace(-offset, offset, filtersize)
Filter = gaussian(Filter, sigma)
Filter = Filter/np.sum(Filter) # Normalize


# ---- Filter the Data ----

# Set Up Output
SmoothedData = np.array(np.zeros(DataLen))

# Left Hand Side
for i in range(0, offset):
    SmoothedData[i] = np.dot(Data[0 : i + offset + 1], Filter[offset - i : filtersize]/np.sum(Filter[offset - i : filtersize]))

# Middle
for i in range(offset, DataLen - offset):
    SmoothedData[i] = np.dot(Data[i - offset : i + offset + 1], Filter)

# Right Hand Side
for i in range(DataLen - offset, DataLen):
    SmoothedData[i] = np.dot(Data[i - offset : DataLen], Filter[0 : filtersize - offset - i + DataLen - 1]/np.sum(Filter[0 : filtersize - offset - i + DataLen - 1]))


# ---- Results ----

# Plotting results
import matplotlib.pyplot as plt

plt.title('Actual Data vs Smoothed Data',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Wind Power Generated')

plt.plot(range(DataLen), Data, 'k', lw = 3)
plt.plot(range(DataLen), SmoothedData, 'b')

plt.show()

Noise = SmoothedData - Data

plt.title('Calculated Noise',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Calculated Noise')

plt.plot(range(DataLen), Noise, 'b')

print 'Noise Mean: ' + str(np.average(Noise))
print 'Noise Std Dev: ' + str(np.std(Noise))

plt.show()


# ---- Loop for Updating Smoothed Data ----
'''
# Note: this would be run when a new data point is received (rather than re-running the whole function for all data)

SmoothedData.append(sum(np.multiply(Data[DataLen - offset - 1 : DataLen], Filter[0 : offset]/sum(Filter[0 : offset]))))

for i in range(DataLen - offset - 1, DataLen - 1):
    SmoothedData[i] = sum(np.multiply(Data[i - offset - 1 : DataLen], Filter[0 : filtersize - offset - i + DataLen]/sum(Filter[0 : filtersize - offset - i +DataLen])))
'''
