# --------------------------------------------------
# Title: BoxFilter1D.py
# Date: 24 June 2014
# Author: Jeremy L Thompson
# Purpose: Implement Box Filter on 1D data
# --------------------------------------------------


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

Data = DataSet[12, 8000 : ]

DataLen = len(Data)


# ---- Set Up Moving Aveage ----

# Get the Moving Average length
window = 7

halfwindow = (window - 1)/2

BoxFilter = np.ones(window)/window


# ---- Filter the Data ----

SmoothedData = np.zeros(len(Data))

# Left Hand Side
for i in range(0, halfwindow):
    SmoothedData[i] = np.sum(Data[0 : i + halfwindow + 1])/(i + halfwindow + 1)

# Middle
for i in  range(halfwindow, DataLen - halfwindow):
    SmoothedData[i] = np.dot(Data[i - halfwindow : i + halfwindow + 1], BoxFilter)

# Right Hand Side
for i in range(DataLen - halfwindow, DataLen):
    SmoothedData[i] = np.sum(Data[i - halfwindow : DataLen])/(DataLen - i + halfwindow)


# ---- Results ----

# Plotting results

import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.suptitle('Real Data Time Series 2 \n Box Filter - $|I| = 7$',  fontsize = 12)
matplotlib.rcParams.update({'font.size': 10})
plt.xlabel('Time Index')
plt.ylabel('Signal')

plt.plot(range(len(Data)), Data, 'b', lw = 2)
plt.plot(range(len(SmoothedData)), SmoothedData, 'r', lw = 2)
plt.xlim([0, DataLen])

fig.set_size_inches(10, 4)
plt.savefig('BoxRealSignal2.png')

plt.show()
