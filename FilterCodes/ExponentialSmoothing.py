# --------------------------------------------------
# Title: ExponentialSmoothing.py
# Date: 10 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement Exponential Smoothing on 1D data
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
Data = np.array([int(i) for i in transArr[3][1:]])

DataLen = len(Data)


# ---- Smooth the Data ----

# Get the alpha value
alpha = float(raw_input("Please enter the desired alpha value: "))

# Smooth the data w/Moving Average
SmoothedData = np.zeros(DataLen)
SmoothedData[0] = Data[0]

for i in  range(1, DataLen):
    SmoothedData[i] = Data[i]*alpha + SmoothedData[i - 1]*(1.0 - alpha)


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
