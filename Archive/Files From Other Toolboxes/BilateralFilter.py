import scipy

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
Data = transArr[3][1:]
Data = [float(i) for i in Data]

# Apply Gaussian Filter
from skimage.filter import denoise_bilateral
SmoothedData = denoise_bilateral(Data)

# Plotting results
import matplotlib.pyplot as plt

plt.title('Actual Data vs Smoothed Data',  fontsize = 22)
plt.xlabel('Time Index')
plt.ylabel('Wind Power Generated')

plt.plot(range(len(Data)), Data, 'k', lw = 3)
plt.plot(range(len(SmoothedData)), SmoothedData, 'b')

plt.show()
