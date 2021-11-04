# --------------------------------------------------
# Title: DenoisingUtility.py
# Date: 31 July 2014
# Author: Jeremy L Thompson
# Purpose: Menu system for denoising utility
# --------------------------------------------------


# -------- Set Up --------

try:
    setupcomplete
except NameError:
    import numpy as np

    Data = []
    SmoothedData = []
    SmoothedData2 = []

    DataPresent = False
    SmoothedDataPresent = False
    SmoothedData2Present = False

    execfile('Methods.py')
    execfile('FileIO.py')

    setupcomplete = True
    print 'Setup Complete'


# -------- Menu 1 --------

def menu1():

    menu = {}
    menu['1'] = '- Enter File Name'
    menu['2'] = '- File Specifications'
    menu['3'] = '- Return'
    menu['4'] = '- Cancel'

    while True:

        options = menu.keys()
        options.sort()

        print '\n-------------------- Input Data --------------------'

        for entry in options:
            print entry, menu[entry]

        selection = raw_input('\nPlease select one: ')

        if selection == '1':
            Data = LoadData()

            DataPresent = True

            SmoothedData = []
            SmoothedData2 = []

            SmoothedDataPresent = False
            SmoothedData2Present = False

        elif selection == '2':
            print '\nData should be in a single column text file, without headers.\n'

        elif selection == '3':
            try:
                return Data, SmoothedData, SmoothedData2, DataPresent, SmoothedDataPresent, SmoothedData2Present
            except NameError:
                break
        elif selection == '4':
            break

        else:
            print '\nUnknown Option Selected\n'


# -------- Menu 21 --------

def menu21(Data):

    menu = {}
    menu['1'] = '- Box Filter'
    menu['2'] = '- Gaussian Filter'
    menu['3'] = '- Bilateral Filter'
    menu['4'] = '- Fourier Transform'
    menu['5'] = '- Wavelet Transform'
    menu['6'] = '- Non-Local Means'
    menu['7'] = '- Return'
    menu['8'] = '- Cancel'

    while True:

        options = menu.keys()
        options.sort()

        print '\n--------------- Denoise Data ---------------'

        for entry in options:
            print entry, menu[entry]

        selection = raw_input('\nPlease select one: ')

        if selection == '1':
            SmoothedData = BoxFilter(Data)

        elif selection == '2':
            SmoothedData = GaussianFilter(Data)

        elif selection == '3':
            SmoothedData = BilateralFilter(Data)

        elif selection == '4':
            SmoothedData = FFTFilter(Data)

        elif selection == '5':
            SmoothedData = WaveletFilter(Data)

        elif selection == '6':
            SmoothedData = NLMeansFilter(Data)

        elif selection == '7':
            try:
                return SmoothedData
            except NameError:
                print '\nCancelled\n'

                break

        elif selection == '8':
            print '\nCancelled\n'

            break

        else:
            print '\nUnknown Option Selected\n'


# -------- Menu 2 --------

def menu2():

    menu = {}
    menu['1'] = '- Denoise Data'
    menu['2'] = '- Denoise Denoised Data'
    menu['3'] = '- Return'
    menu['4'] = '- Cancel'

    while True:

        options = menu.keys()
        options.sort()

        print '\n-------------------- Denoise Data --------------------'

        for entry in options:
            print entry, menu[entry]

        selection = raw_input('\nPlease select one: ')

        if selection == '1':
            SmoothedData = menu21(Data)
            SmoothedDataPresent = True

            SmoothedData2 = []
            SmoothedData2Present = False

        elif selection == '2':
            try:
                SmoothedDataPresent

                SmoothedData2 = menu21(SmoothedData)
                SmoothedData2Present = True

            except NameError:
                print '\nNo Denoised Data\n'

        elif selection == '3':
            try:
                return SmoothedData, SmoothedData2, SmoothedDataPresent, SmoothedData2Present
            except NameError:
                break

        elif selection == '4':
            break

        else:
            print '\nUnknown Option Selected\n'


# -------- Menu 3 --------

def menu3():

    menu = {}
    menu['1'] = '- Plot Data'
    menu['2'] = '- Return'

    while True:

        options = menu.keys()
        options.sort()

        print '\n-------------------- Plot Data --------------------'

        for entry in options:
            print entry, menu[entry]

        selection = raw_input('\nPlease select one: ')

        if selection == '1':
            PlotData()

        elif selection == '2':
            break

        else:
            print '\nUnknown Option Selected\n'


# -------- Menu 4 --------

def menu4():

    menu = {}
    menu['1'] = '- Save Output'
    menu['2'] = '- Return'

    while True:

        options = menu.keys()
        options.sort()

        print '\n-------------------- Save Output --------------------'

        for entry in options:
            print entry, menu[entry]

        selection = raw_input('\nPlease select one: ')

        if selection == '1':
            if SmoothedData2Present:
                SaveSmoothedData2()

            else:
                SaveSmoothedData()

        elif selection == '2':
            break

        else:
            print '\nUnknown Option Selected\n'


# -------- Main Menu --------

mainmenu = {}
mainmenu['1'] = '- Input Data'
mainmenu['2'] = '- Denoise Data'
mainmenu['3'] = '- Plot Data'
mainmenu['4'] = '- Save Output'
mainmenu['5'] = '- Exit'

while True:

    options = mainmenu.keys()
    options.sort()

    print '\n------------------------- Main Menu -------------------------'

    for entry in options:
        print entry, mainmenu[entry]

    selection = raw_input('\nPlease select one: ')

    if selection == '1':
        try:
            Data, SmoothedData, SmoothedData2, DataPresent, SmoothedDataPresent, SmoothedData2Present = menu1()
        except TypeError:
            print '\nCancelled\n'

    elif selection == '2':
        if not DataPresent:
            print '\nNo Data Loaded\n'
        else:
            try:
                SmoothedData, SmoothedData2, SmoothedDataPresent, SmoothedData2Present = menu2()
            except TypeError:
                print '\nCancelled\n'

    elif selection == '3':
        menu3()

    elif selection == '4':
        if not SmoothedDataPresent:
            print '\nNo Smoothed Data\n'
        else:
            menu4()

    elif selection == '5':
        print '\nGoodby!\n'
        break

    else:
        print '\nUnknown Option Selected\n'
