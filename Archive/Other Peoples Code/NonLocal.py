#-----------------------------------------
# Title: NonLocal.py
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
DataTemp = []
DataTemp = np.array([int(i) for i in transArr[3][1:]])

Data = DataTemp[0 : 500]

DataLen = len(Data)

# ---- Set up Gaussian Filter ----

# Get the sigma
print 'The standard deviation of the data is ' + str(np.std(Data))
sigma = float(raw_input("Please enter the desired kernel standard deviation: "))

# Define Gaussian Function
def gaussian(x, sigma):
    return np.exp( -x*x/(2*sigma*sigma) )/(np.sqrt(2*np.pi)*sigma)

# Build Gaussian Filter
#filtersize = 2*int(5*sigma) + 1
filtersize = 2*int(2*sigma) + 1
filtersize = min(filtersize, DataLen - 1) # prevent overrun

offset = (filtersize - 1)/2

print offset

Filter = np.linspace(-offset, offset, filtersize)
Filter = gaussian(Filter, sigma)
FilterNorm = Filter/sum(Filter) # Normalize

print Filter

print FilterNorm


# Define Gaussian Weighted Euclidean Function
# Input: neighborhood vector x1 and neighborhood neighbor x2
def gEuclideanSq(x1,x2,weight): 
    diff = x1 - x2
    return np.sum(diff*diff*weight)    

    #sumSq = 0.0
    #for i in range(len(x1)):
    #    sumSq += ( (x1(i) - x2(i))**2 )*weight(i)
    #return (sumSq**0.5)



# ---- Filter the Data ----

# Local weights
gSSD = np.zeros(shape=(DataLen,DataLen))

for i in range(0,DataLen):
    for j in range(i,DataLen):
        
        x1 = np.array([])
	x2 = np.array([])

        #x1
	# Left Hand Side
	if i < offset:
	    #weight = Filter[offset - i : filtersize]/sum(Filter[offset - i : filtersize])
            
            for t in range(0,offset-i):
                x1 = np.append(x1,0.)  					

            x1 = np.append(x1,Data[0:i+offset+1]) 

	# Right Hand Side
	elif i > (DataLen - offset - 1):
	    #weight = Filter[0 : filtersize - offset - i + DataLen - 1]/sum(Filter[0 : filtersize - offset - i + DataLen - 1])
            x1 = np.append(x1,Data[i - offset : DataLen+1])
            t_end = offset - (DataLen-i-1)
	    for t in range(0,t_end):
		x1 = np.append(x1,[0.])

	# Middle
	else:
	    #weight = FilterNorm
    	    x1 = np.append(x1,Data[i - offset : i + offset + 1])

        # x2
        # Left Hand Side
	if j < offset:
            
            for t in range(0,offset-j):
                x2 = np.append(x2,[0.])  					

            x2 = np.append(x2,Data[0:j+offset+1]) 

	# Right Hand Side
	elif j > (DataLen - offset - 1):
            x2 = np.append(x2,Data[j - offset : DataLen+1])
            t_end = offset - (DataLen-j-1)
	    for t in range(0,t_end):
		x2 = np.append(x2,[0.])

	# Middle
	else:
    	    x2 = np.append(x2,Data[j - offset : j + offset + 1])
  

	weight = FilterNorm
    	gSSD[i][j] = gEuclideanSq(x1,x2,weight) 
        gSSD[j][i] = gSSD[i][j]


# Global weights
W = np.zeros(shape=(DataLen,DataLen))

for i in range(DataLen):
    sumTemp = 0.
    wTemp = np.zeros(DataLen)
    for j in range(DataLen):
        wTemp[j] = np.exp( -gSSD[i][j]/(sigma*sigma) )
        sumTemp += wTemp[j]
         
    W[i,:] = wTemp/sumTemp    

# ---- Results ----

#print gSSD
#print W

NL = np.zeros(DataLen)

for i in range(DataLen):
    sumTemp = 0.
    for j in range(DataLen):
        sumTemp += W[i][j]*Data[j]    

    NL[i] = sumTemp

print NL
#print W[0,:]
#print Data

#for i in range(DataLen):
#    print W[0][i]*Data[i]

# Plotting results
'''
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
'''


