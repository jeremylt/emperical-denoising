# --------------------------------------------------
# Title: FileIO.py
# Date: 31 July 2014
# Author: Jeremy L Thompson
# Purpose: Load, Plot, and Save
# --------------------------------------------------


# -------- Load Data --------

def LoadData():

    # Get Filename
    filename = raw_input('Please enter the filename: ')

    # Read data into a matrix
    TempArray = []

    try:
        f = open(filename, 'r')
    except IOError:
        print '\nFile Not Found\n'
        return []

    i = 0

    for val in f.readlines():
        number = val.strip()
        TempArray.append(val)

    Data = [float(i) for i in TempArray]

    Data = np.array(Data)

    print '\nFile Loaded\n'

    return Data


# -------- Plot Data --------

def PlotData():

    # Set Up Plot
    import matplotlib.pyplot as plt
    import matplotlib as matplotlib

    DataLen = len(Data)

    if SmoothedData2Present:
        plt.title('Noisy Data vs Smoothed Data',  fontsize = 22)

        Noise = SmoothedData2 - Data

        plt.plot(range(DataLen), Data, 'b', lw = 2, label = 'Noisy Data')
        plt.plot(range(DataLen), SmoothedData, 'r', lw = 2, label = 'Denoised Data')
        plt.plot(range(DataLen), SmoothedData2, 'g', lw = 2, label = 'Twice Denoised Data')
        plt.plot(range(DataLen), Noise, 'c', lw = 2, label = 'Residuals')

        plt.xlim(0, DataLen - 1)

        plt.legend().get_frame().set_alpha(0.5)


    elif SmoothedDataPresent:
        plt.title('Noisy Data vs Smoothed Data',  fontsize = 22)

        Noise = SmoothedData - Data

        plt.plot(range(DataLen), Data, 'b', lw = 2, label = 'Noisy Data')
        plt.plot(range(DataLen), SmoothedData, 'r', label = 'Denoised Data')

        plt.plot(range(DataLen), Noise, 'c', lw = 2, label = 'Residuals')

        plt.xlim(0, DataLen - 1)

        plt.legend().get_frame().set_alpha(0.5)

    else:
        plt.title('Noisy Data',  fontsize = 22
)
        plt.plot(range(DataLen), Data, 'b', lw = 2, label = 'Noisy Data')

        plt.xlim(0, DataLen - 1)

        plt.legend().get_frame().set_alpha(0.5)

    plt.xlabel('Time Index')
    plt.ylabel('Signal')

    while True:
        SavePlot = raw_input('Save the plot? (y/n) ')

        if SavePlot == 'y' or SavePlot == 'Y':
            filename = raw_input('Please enter the filename: ')

            fig = plt.gcf()
            fig.set_size_inches(10, 4)
            matplotlib.rcParams.update({'font.size': 10})
            fig.savefig(filename)

            print '\nWaiting for user to close the plot...\n'

            plt.show()

            break

        elif SavePlot == 'n' or SavePlot == 'N':
            print '\nWaiting for user to close the plot...\n'

            plt.show()

            break

        else:
            print '\nUnknow Option Selected\n'


# -------- Save Data --------

def SaveSmoothedData():

    # Get Filename
    filename = raw_input('Please enter the filename: ')

    # Read data into file
    f = open(filename, 'w')

    f.write('Data;DenoisedData\n')

    for i in range(len(SmoothedData)):
        f.write(str(Data[i]) + ';' + str(SmoothedData[i]) + '\n')

    print '\nFile Saved\n'


def SaveSmoothedData2():

    # Get Filename
    filename = raw_input('Please enter the filename: ')

    # Read data into file
    f = open(filename, 'w')

    f.write('Data;DenoisedData;DenoisedData2\n')

    for i in range(len(SmoothedData)):
        f.write(str(Data[i]) + ';' + str(SmoothedData[i]) + ';' + str(SmoothedData2[i]) + '\n')

    print '\nFile Saved\n'
