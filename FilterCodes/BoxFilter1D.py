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


# ---- Set Up Moving Aveage ----

# Get the Moving Average length
window = int(raw_input("Please enter the number of terms to average (odd number): "))

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
import matplotlib.pyplot as plt

plt.title('Actual Data vs Smoothed Data',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Wind Power Generated')

plt.plot(range(len(Data)), Data, 'k', lw = 3)
plt.plot(range(len(SmoothedData)), SmoothedData, 'b')

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

SmoothedData.append(sum(Data[len(Data) - halfwindow : len(Data))/(halfwindow + 1)))

for i in range(len(Data) - halfwindow - 1, len(Data) - 1):
   SmoothedData[i] = sum(Data[i - halfwindow : len(Data)])/(len(Data) - 1 - i + halfwindow)
'''
