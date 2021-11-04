import matplotlib.image as mpimg
import numpy as np


# ---- Data Input ----

# Read data into a matrix
Data = mpimg.imread('Lena.png')
DataLen = len(Data)

# Plotting Image
import matplotlib.pyplot as plt


# ---- Set Up Filter ----

# Get the Moving Average length
window = int(raw_input("Please enter the number of terms to average (odd number): "))
while window > DataLen:
    window = int(raw_input("The number of terms must be an odd number less than " + str(DataLen) + ": "))
halfwindow = (window - 1)/2

# ---- Filter the Data ----

ImgOut = np.zeros(shape = (DataLen, DataLen))

def BoxFilter1D(Data):
    SmoothedData = Data

    # Left Hand Side
    for i in range(0, halfwindow):
        SmoothedData[i] = sum(Data[0 : i + halfwindow + 1])/(i + halfwindow + 1)

    # Middle
    for i in  range(halfwindow, DataLen - halfwindow):
        SmoothedData[i] = sum(Data[i - halfwindow : i + halfwindow + 1])/window

    # Right Hand Side
    for i in range(DataLen - halfwindow, DataLen):
       SmoothedData[i] = sum(Data[i - halfwindow : DataLen])/(DataLen - i + halfwindow)

    return SmoothedData

# Smooth Horizontally
for row in range(DataLen):
    ImgOut[row, :] = BoxFilter1D(Data[row, :])

Data = ImgOut

# Smooth Vertically
for col in range(DataLen):
    ImgOut[:, col] = BoxFilter1D(Data[:, col])

# ---- Results ----

# Plotting results
import matplotlib.pyplot as plt

plt.gray()
plt.imshow(ImgOut)

plt.show()
