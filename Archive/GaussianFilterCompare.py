import numpy as np
import time as time


# ---- Data Input ----

# Read data into a matrix
arr = []
f = open("2010_10_gen.txt", "r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append(numbers)

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

# Extract the 4th (data) row
Data = transArr[3][1 : ]
Data = np.array([float(i) for i in Data])

DataLen = len(Data)


# ---- Set up Gaussian Filter ----

# Get the sigma
print 'The standard deviation of the data is ' + str(np.std(Data))
sigma = float(raw_input("Please enter the desired kernel standard deviation: "))

start = time.time()

# Define Gaussian Function
def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))

# Build Gaussian Filter
filtersize = 2*int(5*sigma) + 1
filtersize = min(filtersize, DataLen - 1) # prevent overrun

offset = (filtersize - 1)/2

Filter = np.linspace(-offset, offset, filtersize)
Filter = gaussian(Filter, sigma)
Filter = Filter/sum(Filter) # Normalize


# ---- Filter the Data ----

# Set Up Output
SmoothedData = np.array(np.zeros(DataLen))

# Left Hand Side
for i in range(0, offset):
    SmoothedData[i] = np.dot(Data[0 : i + offset + 1], Filter[offset - i : filtersize]/sum(Filter[offset - i : filtersize]))

# Middle
for i in range(offset, DataLen - offset):
    SmoothedData[i] = np.dot(Data[i - offset : i + offset + 1], Filter)

# Right Hand Side
for i in range(DataLen - offset, DataLen):
    SmoothedData[i] = np.dot(Data[i - offset : DataLen], Filter[0 : filtersize - offset - i + DataLen - 1]/sum(Filter[0 : filtersize - offset - i + DataLen - 1]))

print 'time 1: ' + str(time.time() - start)

start = time.time()

# Apply Gaussian Filter From Toolbox
from scipy.ndimage.filters import gaussian_filter1d
SmoothedData2 = gaussian_filter1d(Data,sigma)

print 'time 2: ' + str(time.time() - start)

Error = SmoothedData-SmoothedData2

# Plotting Error
import matplotlib.pyplot as plt

plt.title('Error',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Error')

plt.plot(range(len(Error)), Error, 'b')

plt.show()
