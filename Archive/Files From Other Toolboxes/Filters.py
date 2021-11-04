# --------------------------------------------------
# Title: Filters.py
# Date: 29 May 2014
# Author: Jeremy L Thompson
# Purpose: Implement different data smoothing
# techniques and plot the results vs the original
# --------------------------------------------------

# ---------- Importing Data ----------


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


# ----------- Smoothing Functions ----------


def MovingAverage(Data): # Moving Average
    # Get the Moving Average length
    window = int(raw_input("Please enter the number of terms to average: "))

    # Smooth the data w/Moving Average
    SmoothedData = []
    for i in  range(len(Data)-window+1):
        SmoothedData.append(sum(Data[i:i+window-1])/window)

    return SmoothedData, window


def ExponentialSmoothing(Data): # Exponential Smoothing
    # Get the alpha value
    alpha = float(raw_input("Please enter the desired alpha value: "))

    # Smooth the data w/Moving Average
    SmoothedData = []
    SmoothedData.append(Data[0])
    for i in  range(1, len(Data)):
        SmoothedData.append(Data[i]*alpha+SmoothedData[i-1]*(1.0-alpha))

    return SmoothedData, alpha


def GaussianFilter(Data): # Gaussian Filter
    import scipy

    # Get the sigma
    sigma = float(raw_input("Please enter the desired kernal standard deviation: "))

    # Apply Gaussian Filter
    from scipy.ndimage.filters import gaussian_filter1d
    SmoothedData = gaussian_filter1d(Data,sigma)

    return SmoothedData, sigma


def BilateralFilter(Data): # Bilateral Filter
# Apply Gaussian Filter
    from skimage.filter import denoise_bilateral

    SmoothedData = denoise_bilateral(Data)

    return SmoothedData



# ----------- Plotting Results ----------


# Plotting results
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(4)
fig.tight_layout()
fig.suptitle('Actual Data vs Smoothed Data', fontsize=22)
fig.subplots_adjust(top=0.9)


# Plot Moving Average
axarr[0].set_title('Moving Average')
axarr[0].set_ylabel('Wind Power Generated')
axarr[0].set_xlabel('Time Index')

axarr[0].plot(range(len(Data)),Data,'k',lw=3)

MovingAverageData, window = MovingAverage(Data)
axarr[0].set_title('Moving Average - %i'%window)
axarr[0].plot(range(len(MovingAverageData)),MovingAverageData,'r')


# Plot Exponential Smoothing
axarr[1].set_title('Exponential Smoothing')
axarr[1].set_ylabel('Wind Power Generated')
axarr[1].set_xlabel('Time Index')

axarr[1].plot(range(len(Data)),Data,'k',lw=3)

ExponentialSmoothedData, alpha = ExponentialSmoothing(Data)
axarr[1].set_title('Exponential Smoothing - 0.%d'%(alpha*100))
axarr[1].plot(range(len(ExponentialSmoothedData)),ExponentialSmoothedData,'y')


# Plot Gaussian Filter
axarr[2].set_title('Gaussian Filter')
axarr[2].set_ylabel('Wind Power Generated')
axarr[2].set_xlabel('Time Index')

axarr[2].plot(range(len(Data)),Data,'k',lw=3)

GaussianFilterData, sigma = GaussianFilter(Data)
axarr[2].set_title('Gaussian Filter - %d'%sigma)
axarr[2].plot(range(len(GaussianFilterData)),GaussianFilterData,'g')

# Plot Bilateral Filter
axarr[3].set_title('Bilateral Filter')
axarr[3].set_ylabel('Wind Power Generated')
axarr[3].set_xlabel('Time Index')

axarr[3].plot(range(len(Data)),Data,'k',lw=3)

BilateralFilterData = BilateralFilter(Data)
axarr[3].plot(range(len(BilateralFilterData)),BilateralFilterData,'b')


# Show plot
plt.show()

fig.savefig('results.png')
print 'The plot was saved as results.png'


# ---------- End ----------
