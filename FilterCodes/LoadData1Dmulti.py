import numpy as np
import re as re


# ---- Import Data ----

# Read data into a matrix
arr = []
f = open("2011_10.dat","r")
for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append([float(val) for val in numbers])

# Transpose the matrix
transArr = []
transArr = [list(col) for col in zip(*arr)]

Data = np.array(transArr)

dims = np.shape(Data)

# Get Headers

Headers = []

f = open("AttributeNames.txt","r")

for line in f.readlines():
    header = str(line.strip().split('\t'))

    header = re.sub('\W', '', header)
    header = re.sub('_', ' ', header)

    Headers.append(str(header))

# Fix Wind Direction Jumps

def unwrap(Data):
    jump = True

    while jump == True:

        jump = False # stop repeating when no more adjustments

        for i in range(1, len(Data)): # unwrap
            if Data[i - 1] - Data[i] < -180:
                Data[i] -= 360
                jump = True
            elif Data[i - 1] - Data[i] > 180:
                Data[i] += 360
                jump = True

    return Data

for i in range(dims[0]):
    if 'Direction' in Headers[i]:
        Data[i] = unwrap(Data[i])


# ---- Plot ----

# Plotting Time Series
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(57)
fig.suptitle('Actual Data', fontsize = 22)
#fig.subplots_adjust(top = 0.9)
fig.set_size_inches(68, 88)

for i in range(57):
    axarr[i].set_title('Data Set %i' %i)
    axarr[i].plot(range(dims[1]), Data[i])

fig.savefig('MultiDataStreams.png')
print 'The plot was saved as MultiDataStreams.png'
