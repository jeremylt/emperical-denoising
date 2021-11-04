import numpy as np
import time as time
import re as re
start = time.time()


# ---- Data Input ----

# Read data into a matrix

arr = []

f = open("2011_10.dat", "r")

for line in f.readlines():
    numbers = line.strip().split('\t')
    arr.append([float(val) for val in numbers])

# Transpose the matrix

transArr = []
transArr = [list(col) for col in zip(*arr)]

DataSetTemp = np.array(transArr)

DataSet = DataSetTemp[40 : 42, 7900 : 8156 ]

dims = np.shape(DataSet)
DataLen = dims[1]

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

timer = time.time()

for i in range(dims[0]):
    if 'Direction' in Headers[i]:
        DataSet[i] = unwrap(DataSet[i])

timer -= time.time()

print "Unwrapping Time: " + str(-timer)


# -------- Box Filter --------

def BoxFilter(Data):
    # ---- Set Up Moving Average

    window = 5
    halfwindow = (window - 1)/2

    DataLen = len(Data)

    BoxFilter = np.ones(window)/window

    # ---- Filter the Data ----

    BoxFilterData = np.zeros(DataLen)

    # Left Hand Side
    for i in range(0, halfwindow):
        BoxFilterData[i] = np.sum(Data[0 : i + halfwindow + 1])/(i + halfwindow + 1)

    # Middle
    for i in  range(halfwindow, DataLen - halfwindow):
        BoxFilterData[i] = np.dot(Data[i - halfwindow : i + halfwindow + 1], BoxFilter)

    # Right Hand Side
    for i in range(DataLen - halfwindow, DataLen):
        BoxFilterData[i] = np.sum(Data[i - halfwindow : DataLen])/(DataLen - i + halfwindow)

    return BoxFilterData, window


# -------- Gaussian Filter --------

# Define Gaussian Function

def gaussian(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))


def GaussianFilter(Data):

    # ---- Set up Gaussian Filter ----
    sigma = 2.1

    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigma) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    Filter = np.linspace(-offset, offset, filtersize)
    Filter = gaussian(Filter, sigma)
    Filter = Filter/np.sum(Filter) # Normalize


    # ---- Filter the Data ----

    # Set Up Output
    GaussianFilterData = np.array(np.zeros(DataLen))

    # Left Hand Side
    for i in range(0, offset):
        GaussianFilterData[i] = np.dot(Data[0 : i + offset + 1], Filter[offset - i : filtersize]/np.sum(Filter[offset - i : filtersize]))

    # Middle
    for i in range(offset, DataLen - offset):
        GaussianFilterData[i] = np.dot(Data[i - offset : i + offset + 1], Filter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        GaussianFilterData[i] = np.dot(Data[i - offset : DataLen], Filter[0 : filtersize - offset - i + DataLen - 1]/np.sum(Filter[0 : filtersize - offset - i + DataLen - 1]))

    return GaussianFilterData, sigma


# -------- Bilateral Filter --------

def varest(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest


def BilateralFilter(Data):

    # ---- Set Up Gaussian Filter ----
    sigmadist = 2.1
    sigmaint = 2.0*np.sqrt(varest(Data))

    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigmadist) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    SpatialFilter = np.linspace( - offset, offset, filtersize)
    SpatialFilter = gaussian(SpatialFilter, sigmadist)
    SpatialFilter *= 1/np.sum(SpatialFilter) # Normalize


    # ---- Filter the Data ----

    # Set Up Output
    BilateralFilterData = np.zeros(DataLen)

    IntensityFilter = np.array([])

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter
        IntensityFilter = Data[0 : i + offset + 1] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[offset - i : filtersize])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[0 : i + offset + 1], IntensityFilter)

    # Middle
    for i in range(offset, DataLen - offset):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : i + offset + 1] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter)
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : i + offset + 1], IntensityFilter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : DataLen] - Data[i]
        IntensityFilter = gaussian(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen + offset) - 1])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : DataLen], IntensityFilter)

    return BilateralFilterData, sigmadist, sigmaint


# -------- FFT Filter --------

# ---- Slow DFT ----

def SFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(-2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast DFT ----

def FFT(x):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        return SFT(x)
    else:
        X_even = FFT(x[ : : 2])
        X_odd = FFT(x[1 : : 2])
        factor = np.exp(-2j*np.pi*np.arange(N)/N)

        return np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])


# ---- Slow iDFT ----

def iSFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    coeffs = np.exp(2j*np.pi*k*n/N)

    return np.dot(coeffs, x)


# ---- Fast iDFT ----

def iFFT(x, initial):
    N = len(x)

    if N % 2 > 0 or N <= 32:
        output = iSFT(x)
    else:
        X_even = iFFT(x[ : : 2], False)
        X_odd = iFFT(x[1 : : 2], False)
        factor = np.exp(2j*np.pi*np.arange(N)/N)

        output = np.concatenate([X_even + factor[ : N/2]*X_odd, X_even + factor[N/2 : ]*X_odd])

    if initial == True:
        output /= N
        return output
    else:
        return output

def FFTFilter(Data):

    # ---- FFT Data ----

    FTData = FFT(Data)


    # ---- Smooth Data ----

    #softcut = 3/2*np.std(FTData)

    #for i in range(DataLen):
    #    if np.abs(FTData[i]) > softcut:
    #        FTData[i] -= np.sign(FTData[i])*softcut
    #    else:
    #        FTData[i] = 0

    hardcut = 3/2*np.std(FTData)

    for i in range(DataLen):
        if np.abs(FTData[i]) < hardcut:
            FTData[i] = 0


    # ---- Inverse FFT Data ----

    FFTFilterData = np.real(iFFT(FTData, True))

    return FFTFilterData


# -------- Wavelet Filter --------

# ---- DWT ----

def DWT(x):
    N = len(x)

    if N == 1 or N % 2 > 0:
        return x, N

    output = np.zeros(N)

    for i in range(N/2):
        output[i] = (x[i*2] + x[i*2 + 1])/2.
        output[N/2 + i] = (x[i*2] - x[i*2 + 1])/2.

    output[0 : N/2], length = DWT(output[0 : N/2])

    return output, length


# ---- iDWT ----

def iDWT(x, p, length):
    N = len(x)
    cap = 2**p*length

    if cap == N:
        return x

    output = np.zeros(N)
    output[2*cap : N] = x[2*cap : N]

    for i in range(cap):
        output[2*i] = x[i] + x[cap + i]
        output[2*i + 1] = x[i] - x[cap + i]

    output = iDWT(output, p + 1, length)

    return output


def WaveletFilter(Data):

    # ---- DWT Data ----

    WTData, length = DWT(Data)


    # ---- Smooth Data ----

    #softcut = 3/2*np.std(WTData)

    #for i in range(DataLen):
    #    if np.abs(WTData[i]) > softcut:
    #        WTData[i]  -= np.sign(WTData[i])*softcut
    #    else:
    #        WTData[i] = 0

    hardcut = 3/2*np.std(WTData)

    for i in range(DataLen):
        if np.abs(WTData[i]) < hardcut:
            WTData[i] = 0


    # ---- Inverse DWT Data ----

    WaveletFilterData = iDWT(WTData, 0, length)

    return WaveletFilterData


# ---- NL Means ----

def NLMeansFilter(Data):

    # ---- Set up NL Means Filter ----

    DataLen = len(Data)
    IntensityFilter = np.zeros(DataLen)

    beta = 1.0
    offset = 3
    thresh = 30
    var = varest(Data)


    # ---- Filter the Data ----

    # Set Up Output
    SmoothedData = np.zeros(DataLen)

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter

        for j in range(i, DataLen - offset):
            norm = np.sum((Data[0 : i + offset + 1] - Data[j - i : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

        IntensityFilter = np.zeros(DataLen)

    # Middle
    IntensityFilter = np.zeros(DataLen)

    for i in range(offset, DataLen - offset):
        # Update Intensity Filter

        for j in range(offset, DataLen - offset):
            norm = np.sum((Data[i - offset : i + offset + 1] - Data[j - offset : j + offset + 1])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    # Right Hand Side
    IntensityFilter = np.zeros(DataLen)

    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter

        for j in range(offset, DataLen - (DataLen - i)):
            norm = np.sum((Data[i - offset : DataLen] - Data[j - offset : j + (DataLen - i)])**2)
            if norm < thresh:
                IntensityFilter[j] = np.exp(-norm/(2*beta*var*(2*offset + 1)))
            else:
                IntensityFilter[j] = 0

        IntensityFilter[i] = 1

        IntensityFilter *= 1/np.sum(IntensityFilter)

        # Smooth Next Data Point
        SmoothedData[i] = np.dot(Data, IntensityFilter)

    return SmoothedData, beta, (2*offset + 1)


# ---- Exponential Smoothing ----

def ExponentialSmoothing(Data):

    # ---- Set Up Alpha ----
    alpha = 0.25

    DataLen = len(Data)

    SmoothedData = np.zeros(DataLen)

    SmoothedData[0] = Data[0]

    for i in range(1, DataLen):
        SmoothedData[i] = SmoothedData[i - 1]*alpha + Data[i]*(1.0 - alpha)

    return SmoothedData, alpha


# ---- Results ----

# Plotting results
import matplotlib.pyplot as plt

for i in range(dims[0]):

    print float(i)/dims[0]

    Data = DataSet[i]
    datamean = np.average(Data)
    Data -= datamean

    # Setup Figure
    plt.close('all')
    fig, axarr = plt.subplots(7)
    fig.tight_layout()
    fig.suptitle('Actual Data vs Smoothed Data', fontsize = 22)
    fig.subplots_adjust(top = 0.9)
    fig.set_size_inches(22, 17)


    # Plot Moving Average

    timer = time.time()
    BoxFilterData, window = BoxFilter(Data)
    timer -= time.time()

    axarr[0].set_title('Moving Average \n Window Size - %i \n Time - %f' %(window, -timer))
    axarr[0].set_ylabel('%s' %Headers[i])
    axarr[0].set_xlabel('Time Index')

    axarr[0].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[0].plot(range(DataLen), BoxFilterData + datamean, 'r')
    axarr[0].set_xlim([0, DataLen])


    # Plot Gaussian Filter

    timer = time.time()
    GaussianFilterData, sigma = GaussianFilter(Data)
    timer -= time.time()

    axarr[1].set_title('Gaussian Filter \n Temporal Sigma - %f \n Time - %f' %(sigma, -timer))
    axarr[1].set_ylabel('%s' %Headers[i])
    axarr[1].set_xlabel('Time Index')

    axarr[1].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[1].plot(range(DataLen), GaussianFilterData + datamean, 'y')
    axarr[1].set_xlim([0, DataLen])


    # Plot Bilateral Filter

    timer = time.time()
    BilateralFilterData, sigmadist, sigmaint = BilateralFilter(Data)
    timer -= time.time()

    axarr[2].set_title('Bilateral Filter \n Temporal Sigma - %f  Spatial Sigma - %f \n Time - %f' %(sigmadist, sigmaint, -timer))
    axarr[2].set_ylabel('%s' %Headers[i])
    axarr[2].set_xlabel('Time Index')

    axarr[2].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[2].plot(range(DataLen), BilateralFilterData + datamean, 'g')
    axarr[2].set_xlim([0, DataLen])


    # Plot FFT Filter

    timer = time.time()
    FFTFilterData = FFTFilter(Data)
    timer -= time.time()

    axarr[3].set_title('FFT Filter \n Time - %f' %(-timer))
    axarr[3].set_ylabel('%s' %Headers[i])
    axarr[3].set_xlabel('Time Index')

    axarr[3].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[3].plot(range(DataLen), FFTFilterData + datamean, 'c')
    axarr[3].set_xlim([0, DataLen])


    # Plot Wavelet Filter

    timer = time.time()
    WaveletFilterData = WaveletFilter(Data)
    timer -= time.time()

    axarr[4].set_title('Wavelet Filter \n Time - %f' %(-timer))
    axarr[4].set_ylabel('%s' %Headers[i])
    axarr[4].set_xlabel('Time Index')

    axarr[4].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[4].plot(range(DataLen), WaveletFilterData + datamean, 'b')
    axarr[4].set_xlim([0, DataLen])


    # Plot NL Means Filter

    timer = time.time()
    NLMeansFilterData, beta, window = NLMeansFilter(Data)
    timer -= time.time()

    axarr[5].set_title('NL Means Filter \n Beta - %f  Window Size - %i \n Time - %f' %(beta, window, -timer))
    axarr[5].set_ylabel('%s' %Headers[i])
    axarr[5].set_xlabel('Time Index')

    axarr[5].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[5].plot(range(DataLen), NLMeansFilterData + datamean, 'm')
    axarr[5].set_xlim([0, DataLen])


    # Plot Exponential Smoothing

    timer = time.time()
    ExponentialSmoothingData, alpha = ExponentialSmoothing(Data)
    timer -= time.time()

    axarr[6].set_title('Exponential Smoothing \n Alpha - 0.%d \n Time - %f' %(alpha*100, -timer))
    axarr[6].set_ylabel('%s' %Headers[i])
    axarr[6].set_xlabel('Time Index')

    axarr[6].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[6].plot(range(DataLen), ExponentialSmoothingData + datamean, 'r')
    axarr[6].set_xlim([0, DataLen])


    # Show plot
    #plt.show()

    fig.savefig('results%d.png' %i)


    # Setup Figure
    plt.close('all')
    fig, axarr = plt.subplots(7)
    fig.tight_layout()
    fig.suptitle('Error', fontsize = 22)
    fig.subplots_adjust(top = 0.9)
    fig.set_size_inches(22, 17)


    # Plot Moving Average Error
    axarr[0].set_title('Moving Average')
    axarr[0].set_ylabel('%s Error' %Headers[i])
    axarr[0].set_xlabel('Time Index')

    axarr[0].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[0].plot(range(DataLen), BoxFilterData - Data, 'r')
    axarr[0].set_xlim([0, DataLen])


    # Plot Gaussian Filter Error
    axarr[1].set_title('Gaussian Filter')
    axarr[1].set_ylabel('%s Error' %Headers[i])
    axarr[1].set_xlabel('Time Index')

    axarr[1].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[1].plot(range(DataLen), GaussianFilterData - Data, 'y')
    axarr[1].set_xlim([0, DataLen])


    # Plot Bilateral Filter Error
    axarr[2].set_title('Bilateral Filter')
    axarr[2].set_ylabel('%s Error' %Headers[i])
    axarr[2].set_xlabel('Time Index')

    axarr[2].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[2].plot(range(DataLen), BilateralFilterData - Data, 'g')
    axarr[2].set_xlim([0, DataLen])


    # Plot FFT Filter Error
    axarr[3].set_title('FFT Filter')
    axarr[3].set_ylabel('%s Error' %Headers[i])
    axarr[3].set_xlabel('Time Index')

    axarr[3].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[3].plot(range(DataLen), FFTFilterData - Data, 'c')
    axarr[3].set_xlim([0, DataLen])


    # Plot Wavelet Filter Error
    axarr[4].set_title('Wavelet Filter')
    axarr[4].set_ylabel('%s Error' %Headers[i])
    axarr[4].set_xlabel('Time Index')

    axarr[4].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[4].plot(range(DataLen), WaveletFilterData - Data, 'b')
    axarr[4].set_xlim([0, DataLen])


    # Plot NL Means Filter Error
    axarr[5].set_title('NL Means Filter')
    axarr[5].set_ylabel('%s Error' %Headers[i])
    axarr[5].set_xlabel('Time Index')

    axarr[5].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[5].plot(range(DataLen), NLMeansFilterData - Data, 'm')
    axarr[5].set_xlim([0, DataLen])


    # Plot Exponential Smoothing Error
    axarr[6].set_title('Exponential Smoothing')
    axarr[6].set_ylabel('%s Error' %Headers[i])
    axarr[6].set_xlabel('Time Index')

    axarr[6].plot(range(DataLen), Data + datamean, 'k', lw = 2)

    axarr[6].plot(range(DataLen), ExponentialSmoothingData - Data, 'r')
    axarr[6].set_xlim([0, DataLen])


    # Show plot
    #plt.show()

    fig.savefig('errorresults%d.png' %i)

print 'Total Time: ' + str(time.time() - start)
