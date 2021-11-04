import matplotlib.image as mpimg
import numpy as np

# ---- Data Input ----

# Read data into a matrix
Data = mpimg.imread('Lena.png')
DataLen = len(Data)


# ---- Set up Gaussian Filter ----

# Get the sigma
sigma = float(raw_input("Please enter the desired kernel standard deviation: "))

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

ImgOut = np.zeros(shape = (DataLen, DataLen))

def GaussianFilter1D(Data):
    SmoothedData = Data

    # Left Hand Side
    for i in range(0, offset):
        SmoothedData[i] = np.dot(Data[0 : i + offset + 1], Filter[offset - i : filtersize]/np.sum(Filter[offset - i : filtersize]))

    # Middle
    for i in range(offset, DataLen - offset):
        SmoothedData[i] = np.dot(Data[i - offset : i + offset + 1], Filter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        SmoothedData[i] = np.dot(Data[i - offset : DataLen], Filter[0 : filtersize - offset - i + DataLen - 1]/np.sum(Filter[0 : filtersize - offset - i + DataLen - 1]))

    return SmoothedData

# Smooth Horizontally
for row in range(DataLen):
    ImgOut[row, :] = GaussianFilter1D(Data[row, :])

Data = ImgOut

# Smooth Vertically
for col in range(DataLen):
    ImgOut[:, col] = GaussianFilter1D(Data[:, col])

# ---- Results ----

# Plotting results
import matplotlib.pyplot as plt

plt.gray()
plt.imshow(ImgOut)

plt.show()
