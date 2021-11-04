import numpy as np
import random as rnd


# -------- Generate Synthetic Signal 2 --------

def buildsignal2(pernoise):

    DataLen = 256
    TrueData = np.zeros(DataLen)
    Data = np.zeros(DataLen)

    # ---- Set Parameters ----

    mu = 25.0
    amp1 = 10.0
    amp2 = 2.0
    per1 = 712.0
    per2 = 128.0
    jump = 12.0

    noisesig = (amp1 + amp2 + jump)*pernoise/100

    # ---- Build Signals ----

    for i in range(DataLen/3, DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(2*DataLen/3, 2*DataLen/3+DataLen/100):
        TrueData[i] += jump
        Data[i] += jump

    for i in range(DataLen):
        TrueData[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2)
        Data[i] += mu + amp1*np.sin(i*2*np.pi/per1) + amp2*np.sin(i*2*np.pi/per2) + rnd.gauss(0, noisesig)

    return TrueData, Data


# ---- Load Methods ----

execfile('Methods.py')


# ---- Results ----

DataLen = 256
TrueData, Data = buildsignal2(10)

# Plotting results
import matplotlib as matplotlib
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(6)
fig.suptitle('Simulated Data - 10% Noise - Time Series 2',  fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.set_size_inches(10, 8)
matplotlib.rcParams.update({'font.size': 10})


# Plot Box Filter
SmoothedData = BoxFilter(Data, 7)

axarr[0].set_title('Box Filter - $|I|$ = 7')
axarr[0].set_xlabel('Time Index')
axarr[0].set_ylabel('Signal')

axarr[0].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[0].plot(range(DataLen), Data, 'b', lw = 2)
axarr[0].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[0].set_xlim(0, DataLen)


# Plot Gaussian Filter
SmoothedData = GaussianFilter(Data, 2.25)

axarr[1].set_title('Gaussian Filter - $\sigma_d$ = 2.25')
axarr[1].set_xlabel('Time Index')
axarr[1].set_ylabel('Signal')

axarr[1].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[1].plot(range(DataLen), Data, 'b', lw = 2)
axarr[1].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[1].set_xlim(0, DataLen)


# Plot Bilateral Filter
SmoothedData = BilateralFilter(Data, 2.5, 2.5)

axarr[2].set_title('Bilateral Filter - $\sigma_d$ = 2.5 - $\sigma_i$ = 2.5 $\hat{\sigma}_n$')
axarr[2].set_xlabel('Time Index')
axarr[2].set_ylabel('Signal')

axarr[2].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[2].plot(range(DataLen), Data, 'b', lw = 2)
axarr[2].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[2].set_xlim(0, DataLen)


# Plot FFT Filter
SmoothedData = FFTFilter(Data, 0.15)

axarr[3].set_title('Fourier Transform Coefficient Thresholding - $\mu$ = 0.15 $\sigma$')
axarr[3].set_xlabel('Time Index')
axarr[3].set_ylabel('Signal')

axarr[3].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[3].plot(range(DataLen), Data, 'b', lw = 2)
axarr[3].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[3].set_xlim(0, DataLen)


# Plot Wavelet Filter
SmoothedData = WaveletFilter(Data, 0.3)

axarr[4].set_title('Wavelet Transform Coefficient Thresholding - $\mu$ = 0.3 $\sigma$')
axarr[4].set_xlabel('Time Index')
axarr[4].set_ylabel('Signal')

axarr[4].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[4].plot(range(DataLen), Data, 'b', lw = 2)
axarr[4].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[4].set_xlim(0, DataLen)


# Plot Non-Local Means Filter
SmoothedData = NLMeansFilter(Data, 0.5)

axarr[5].set_title('Non-Local Means Filter - $\\beta$ = 0.5, $|I|$ = 7, $T$ = 0.75 $( \mathrm{max} Y_j - \mathrm{min} Y_j )$')
axarr[5].set_xlabel('Time Index')
axarr[5].set_ylabel('Signal')

axarr[5].plot(range(DataLen), TrueData, 'k', lw = 2)
axarr[5].plot(range(DataLen), Data, 'b', lw = 2)
axarr[5].plot(range(DataLen), SmoothedData, 'r', lw = 2)
axarr[5].set_xlim(0, DataLen)


fig.savefig('TimeSeries2SimulatedCompare.png')
plt.show()
