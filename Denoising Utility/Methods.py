# --------------------------------------------------
# Title: Methods.py
# Date: 31 July 2014
# Author: Jeremy L Thompson
# Purpose: All denosing methods as functions
# --------------------------------------------------


# -------- Helper Functions --------

# Define Gaussian Function

def GaussianFunc(x, sigma):
    return np.exp(-np.power(x, 2.)/(2*np.power(sigma, 2.)))


# Define Variance Estimator

def VarEst(Data):
    DataLen = len(Data)

    varest = 0.0

    for i in range(1, DataLen - 1):
        varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

    varest *= 2.0/(3.0*(DataLen - 2))

    return varest


# -------- Box Filter --------

def BoxFilter(Data):

    # ---- Set Up Moving Average

    window = int(raw_input('Please enter the number of terms to average (odd integer): '))
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

    print '\nData Denoised\n'

    return BoxFilterData


# -------- Gaussian Filter --------

def GaussianFilter(Data):

    # ---- Set up Gaussian Filter ----

    sigma = float(raw_input('Please enter the spatial kernel standard deviation (positive number): '))
    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigma) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    Filter = np.linspace(-offset, offset, filtersize)
    Filter = GaussianFunc(Filter, sigma)
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

    print '\nData Denoised\n'

    return GaussianFilterData


# -------- Bilateral Filter --------

def BilateralFilter(Data):

    # ---- Set Up Gaussian Filter ----

    sigmadist = float(raw_input('Please enter the spatial kernel standard deviation (positive number): '))
    sigintfac = float(raw_input('Please enter the scaling factor for the intensity kernal standard deviation (positive number): '))

    sigmaint = sigintfac*np.sqrt(VarEst(Data))

    DataLen = len(Data)

    # Build Gaussian Filter
    filtersize = 2*int(5*sigmadist) + 1
    filtersize = min(filtersize, DataLen - 1) # prevent overrun

    offset = (filtersize - 1)/2

    SpatialFilter = np.linspace( - offset, offset, filtersize)
    SpatialFilter = GaussianFunc(SpatialFilter, sigmadist)
    SpatialFilter *= 1/np.sum(SpatialFilter) # Normalize


    # ---- Filter the Data ----

    # Set Up Output
    BilateralFilterData = np.zeros(DataLen)

    IntensityFilter = np.array([])

    # Left Hand Side
    for i in range(0, offset):
        # Update Intensity Filter
        IntensityFilter = Data[0 : i + offset + 1] - Data[i]
        IntensityFilter = GaussianFunc(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[offset - i : filtersize])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[0 : i + offset + 1], IntensityFilter)

    # Middle
    for i in range(offset, DataLen - offset):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : i + offset + 1] - Data[i]
        IntensityFilter = GaussianFunc(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter)
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : i + offset + 1], IntensityFilter)

    # Right Hand Side
    for i in range(DataLen - offset, DataLen):
        # Update Intensity Filter
        IntensityFilter = Data[i - offset : DataLen] - Data[i]
        IntensityFilter = GaussianFunc(IntensityFilter, sigmaint)
        IntensityFilter = np.multiply(IntensityFilter, SpatialFilter[0 : filtersize - (i - DataLen + offset) - 1])
        IntensityFilter *= 1/np.sum(IntensityFilter) # Normalize

        # Smooth Next Data Point
        BilateralFilterData[i] = np.dot(Data[i - offset : DataLen], IntensityFilter)

    print '\nData Denoised\n'

    return BilateralFilterData


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

    # ---- Set Up FFT Filter ----

    cutofffactor = float(raw_input('Please enter the hard threshold factor (positive number): '))

    datamean = np.average(Data)

    DataLen = len(Data)

    Data -= datamean

    # ---- FFT Data ----

    FTData = FFT(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(FTData)

    for i in range(DataLen):
        if np.abs(FTData[i]) < hardcut:
            FTData[i] = 0


    # ---- Inverse FFT Data ----

    FFTFilterData = np.real(iFFT(FTData, True))

    Data += datamean
    FFTFilterData += datamean

    print '\nData Denoised\n'

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

    # ---- Set Up Wavelet Filter ----

    cutofffactor = float(raw_input('Please enter the hard threshold factor (positive number): '))

    datamean = np.average(Data)

    DataLen = len(Data)

    Data -= datamean

    # ---- DWT Data ----

    WTData, length = DWT(Data)


    # ---- Smooth Data ----

    hardcut = cutofffactor*np.std(WTData)

    for i in range(DataLen):
        if np.abs(WTData[i]) < hardcut:
            WTData[i] = 0


    # ---- Inverse DWT Data ----

    WaveletFilterData = iDWT(WTData, 0, length)

    Data += datamean

    WaveletFilterData += datamean

    print '\nData Denoised\n'

    return WaveletFilterData


# ---- NL Means ----

def NLMeansFilter(Data):

    # ---- Set up NL Means Filter ----

    DataLen = len(Data)
    IntensityFilter = np.zeros(DataLen)

    window = int(raw_input('Please enter the window size (odd integer): '))
    beta = float(raw_input('Please enter beta [0, 1]: '))
    tfact = float(raw_input('Please enter delta [0, 1]: '))

    offset = (window - 1)/2
    thresh = (max(Data) - min(Data))*(2*offset + 1)*tfact
    thresh *= thresh
    var = VarEst(Data)


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

    print '\nData Denoised\n'

    return SmoothedData
