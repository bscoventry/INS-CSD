#--------------------------------------------------------------------------------------------------------------------------------------------
# Authors: Brandon S Coventry            Wisconsin Institute for Translational Neuroengineering
# Date: 04/21/2024                       Wisconsin is now fluctuating between winter and spring. Hopefully spring now?
# Purpose: This is a batch function using SPyke to perform 2D-CSD
# Interface: At the moment, this uses TDT based reading and analysis, but can be extended to other systems.
# Revision History: Will be tracked in Github.
# Notes: N/A
#--------------------------------------------------------------------------------------------------------------------------------------------
# Imports go here. Global use imports will be put here, we will upload specifics as needed.
import numpy as np
import tdt              #For reading in tdt files
import matplotlib.pyplot as plt
import dask
from SPyke import Spike_Processed
import pdb
import matlab.engine
import pandas as pd
import scipy.io as sio
from scipy.signal import sosfiltfilt
from kcsd import KCSD2D
saveKW = 'LFPCSD'
precurser = 'Z://PhDData//INSData//'
dataPath = ['INS2102//02_15_21//INS_5PU_0_5PW_1ISI','INS2102//02_15_21//INS_5PU_10PW_5ISI','INS2102//02_16_21//INS_5PU_0_2PW_0_1ISI',
            'INS2102//02_16_21//INS_5PU_0_5PW_0_2ISI','INS2102//02_16_21//INS_5PU_0_7PW_0_5ISI','INS2102//02_16_21//INS_5PU_1PW_0_5ISI','INS2102//02_16_21//INS_5PU_5PW_5ISI',
            'INS2102//02_16_21//INS_5PU_5PW_100ISI','INS2102//02_16_21//INS_5PU_10PW_5ISI','INS2102//02_19_21//INS_5PU_10PW_100ISI','INS2102//02_19_21//INS_50PU_0_5PW_0_2ISI',
            'INS2102//02_22_21//INS_5PU_10PW_50ISI_2','INS2102//02_22_21//INS_10PU_0_2PW_0_5ISI','INS2102//02_22_21//INS_10PU_0_2PW_1ISI','INS2102//02_22_21//INS_10PU_0_5PW_1ISI',
            'INS2102//02_22_21//INS_10PU_0_7PW_0_5ISI','INS2102//02_22_21//INS_10PU_1PW_1ISI','INS2102//02_22_21//INS_10PU_5PW_5ISI','INS2102//02_22_21//INS_20PU_0_2PW_0_5ISI',
            'INS2102//02_25_21//INS_5PU_0_7PW_0_5ISI','INS2102//02_25_21//INS_5PU_10PW_50ISI_4','INS2102//02_26_21//INS_5PU_10PW_50ISI','INS2007//08_21_20//INS_5PU_0_2PW_1ISI',
            'INS2007//08_21_20//INS_5PU_0_2PW_5ISI','INS2007//08_21_20//INS_5PU_0_5PW_1ISI','INS2007//08_21_20//INS_5PU_0_5PW_5ISI','INS2007//08_21_20//INS_5PU_1PW_1ISI',
            'INS2007//08_21_20//INS_5PU_1PW_5ISI','INS2007//08_21_20//INS_5PU_0_2PW_1ISI','INS2007//08_21_20//INS_5PU_5PW_5ISI','INS2007//08_24_20//INS_5PU_0_1PW_5ISI',
            'INS2007//08_24_20//INS_5PU_0_2PW_5ISI','INS2007//08_24_20//INS_5PU_0_5PW_5ISI','INS2007//08_24_20//INS_5PU_1PW_5ISI_2','INS2007//08_24_20//INS_5PU_5PW_5ISI','INS2007//08_26_20//INS_5PU_0_2PW_10ISI',
            'INS2007//08_26_20//INS_5PU_0_2PW_100ISI','INS2007//08_26_20//INS_5PU_0_5PW_10ISI','INS2007//08_26_20//INS_5PU_0_5PW_100ISI','INS2007//08_26_20//INS_5PU_1PW_10ISI',
            'INS2007//08_26_20//INS_5PU_1PW_100ISI','INS2007//08_26_20//INS_5PU_5PW_10ISI','INS2007//08_26_20//INS_5PU_5PW_100ISI','INS2007//08_31_20//INS_5PU_0_2PW_5ISI','INS2007//08_31_20//INS_5PU_0_5PW_5ISI',
            'INS2007//08_31_20//INS_5PU_1PW_5ISI_2','INS2007//08_31_20//INS_5PU_1PW_100ISI','INS2007//08_31_20//INS_5PU_5PW_5ISI','INS2007//08_31_20//INS_5PU_5PW_100ISI','INS2007//08_31_20//INS_5PU_10PW_5ISI',
            'INS2007//09_02_20//INS_5PU_0_5PW_20ISI','INS2007//09_02_20//INS_5PU_1PW_1ISI_3','INS2007//09_02_20//INS_5PU_5PW_1ISI','INS2007//09_02_20//INS_5PU_5PW_50ISI','INS2007//09_02_20//INS_5PU_10PW_50ISI',
            'INS2007//09_02_20//INS_5PU_10PW_100ISI','INS2007//09_03_20//INS_5PU_0_1PW_5ISI','INS2007//09_03_20//INS_5PU_0_2PW_5ISI','INS2007//09_03_20//INS_5PU_1PW_1ISI','INS2007//09_03_20//INS_5PU_10PW_50ISI_3',
            'INS2007//09_09_20//INS_5PU_1PW_1ISI','INS2007//09_09_20//INS_5PU_1PW_5ISI','INS2007//09_09_20//INS_5PU_1PW_10ISI','INS2007//09_09_20//INS_5PU_1PW_50ISI','INS2008//08_22_20//INS_5PU_0_2PW_5ISI',
            'INS2008//08_22_20//INS_5PU_0_5PW_5ISI','INS2008//08_22_20//INS_5PU_1PW_1ISI','INS2008//08_22_20//INS_5PU_1PW_5ISI_2','INS2008//08_22_20//INS_5PU_5PW_1ISI','INS2008//08_22_20//INS_5PU_5PW_5ISI_2',
            'INS2008//08_25_20//INS_5PU_0_2PW_100ISI','INS2008//08_25_20//INS_5PU_0_5PW_100ISI','INS2008//08_25_20//INS_5PU_1PW_100ISI','INS2008//08_25_20//INS_5PU_5PW_100ISI','INS2008//08_26_20_2//INS_5PU_0_2PW_5ISI',
            'INS2008//08_26_20_2//INS_5PU_0_2PW_10ISI','INS2008//08_26_20_2//INS_5PU_0_5PW_5ISI','INS2008//08_26_20_2//INS_5PU_0_5PW_10ISI','INS2008//08_26_20_2//INS_5PU_1PW_5ISI',
            'INS2008//08_26_20_2//INS_5PU_1PW_10ISI','INS2008//08_26_20_2//INS_5PU_5PW_5ISI','INS2008//08_26_20_2//INS_5PU_5PW_10ISI','INS2008//09_01_20//INS_5PU_0_2PW_1ISI','INS2008//09_01_20//INS_5PU_0_5PW_1ISI',
            'INS2008//09_01_20//INS_5PU_1PW_1ISI','INS2008//09_01_20//INS_5PU_5PW_1ISI','INS2008//09_01_20//INS_5PU_5PW_5ISI_2','INS2008//09_01_20//INS_5PU_10PW_50ISI','INS2008//09_08_20//INS_5PU_0_1PW_5ISI',
            'INS2008//09_08_20//INS_5PU_0_2PW_5ISI','INS2008//09_08_20//INS_5PU_0_5PW_20ISI','INS2008//09_08_20//INS_5PU_1PW_5ISI','INS2008//09_08_20//INS_5PU_1PW_20ISI','INS2008//09_08_20//INS_5PU_5PW_5ISI',
            'INS2008//09_08_20//INS_5PU_10PW_50ISI_3','INS2013//10_31_20//INS_5PU_1PW_5ISI','INS2013//10_31_20//INS_5PU_0_5PW_5ISI_2','INS2013//10_31_20//INS_5PU_5PW_5ISI','INS2013//10_31_20//INS_5PU_5PW_100ISI_LaserOn',
            'INS2013//11_01_20//INS_5PU_1PW_1ISI','INS2013//11_01_20//INS_5PU_1PW_5ISI','INS2013//11_01_20//INS_5PU_5PW_1ISI','INS2013//11_01_20//INS_5PU_5PW_5ISI','INS2013//11_01_20//INS_5PU_10PW_50ISI',
            'INS2013//11_02_20//INS_5PU_0_5PW_0_3ISI','INS2013//11_02_20//INS_5PU_0_5PW_1ISI','INS2013//11_02_20//INS_5PU_5PW_1ISI','INS2013//11_02_20//INS_5PU_10PW_25ISI','INS2013//11_02_20//INS_5PU_10PW_50ISI_2',
            'INS2013//11_02_20//INS_10PU_0_5PW_0_25ISI','INS2013//11_04_20//INS_1PU_0_2PW_1ISI','INS2013//11_04_20//INS_1PU_0_5PW_1ISI','INS2013//11_04_20//INS_1PU_5PW_50ISI','INS2013//11_04_20//INS_1PU_10PW_50ISI',
            'INS2013//11_04_20//INS_5PU_10PW_50ISI','INS2013//11_04_20//INS_10PU_0_2PW_1ISI','INS2013//11_07_20//INS_5PU_0_2PW_0_2ISI','INS2013//11_07_20//INS_5PU_0_5PW_0_5ISI','INS2013//11_07_20//INS_5PU_1PW_5ISI',
            'INS2013//11_07_20//INS_5PU_10PW_50ISI','INS2013//11_07_20//INS_10PU_1PW_1ISI','INS2013//11_09_20//INS_5PU_0_2PW_0_5ISI','INS2013//11_09_20//INS_5PU_0_5PW_0_5ISI','INS2013//11_09_20//INS_5PU_1PW_0_5ISI',
            'INS2013//11_09_20//INS_10PU_1PW_1ISI_2','INS2013//11_10_20//INS_1PU_0_2PW_1ISI','INS2013//11_10_20//INS_1PU_0_5PW_1ISI','INS2013//11_10_20//INS_1PU_1PW_1ISI','INS2013//11_10_20//INS_1PU_5PW_1ISI','INS2013//11_10_20//INS_1PU_10PW_1ISI',
            'INS2013//11_12_20//INS_10PU_0_2PW_0_5ISI',
            'INS2013//11_12_20//INS_10PU_0_5PW_1ISI','INS2013//11_12_20//INS_10PU_1PW_5ISI','INS2013//11_12_20//INS_10PU_5PW_5ISI','INS2013//11_12_20//INS_10PU_10PW_10ISI',
            'INS2013//11_29_20//INS_5PU_1PW_1ISI','INS2013//11_29_20//INS_5PU_5PW_5ISI','INS2013//11_29_20//INS_5PU_10PW_5ISI','INS2013//11_29_20//INS_5PU_10PW_25ISI','INS2013//11_29_20//INS_5PU_10PW_50ISI',
            'INS2015//11_26_20//INS_1PU_10PW_5ISI','INS2015//11_26_20//INS_5PU_5PW_5ISI','INS2015//11_26_20//INS_5PU_10PW_50ISI','INS2015//11_26_20//INS_10PU_0_5PW_1ISI','INS2015//11_26_20//INS_10PU_1PW_1ISI',
            'INS2015//11_26_20//INS_10PU_1PW_5ISI','INS2015//11_28_20//INS_5PU_1PW_5ISI','INS2015//11_28_20//INS_5PU_5PW_5ISI','INS2015//11_28_20//INS_5PU_10PW_5ISI','INS2015//11_28_20//INS_5PU_10PW_10ISI',
            'INS2015//11_28_20//INS_5PU_10PW_25ISI_2','INS2015//11_28_20//INS_5PU_10PW_50ISI','INS2015//11_30_20//INS_1PU_0_2PW_5ISI','INS2015//11_30_20//INS_1PU_0_5PW_5ISI','INS2015//11_30_20//INS_1PU_1PW_5ISI',
            'INS2015//11_30_20//INS_1PU_5PW_5ISI','INS2015//11_30_20//INS_1PU_10PW_5ISI','INS2015//11_30_20//INS_5PU_10PW_50ISI','INS2015//12_02_20//INS_5PU_0_1PW_5ISI','INS2015//12_02_20//INS_5PU_0_1PW_50ISI',
            'INS2015//12_02_20//INS_5PU_0_2PW_50ISI','INS2015//12_02_20//INS_5PU_0_5PW_50ISI','INS2015//12_02_20//INS_5PU_0_8PW_50ISI','INS2015//12_02_20//INS_5PU_1PW_50ISI','INS2015//12_02_20//INS_5PU_5PW_50ISI'#,
            'INS2015//12_02_20//INS_5PU_10PW_50ISI','INS2015//12_03_20//INS_5PU_1PW_1ISI','INS2015//12_03_20//INS_5PU_1PW_5ISI','INS2015//12_03_20//INS_5PU_5PW_1ISI','INS2015//12_03_20//INS_5PU_5PW_5ISI',
            'INS2015//12_03_20//INS_5PU_5PW_10ISI','INS2015//12_03_20//INS_5PU_10PW_5ISI','INS2015//12_03_20//INS_5PU_10PW_10ISI','INS2015//12_14_20//INS_5PU_0_5PW_1ISI','INS2015//12_14_20//INS_5PU_5PW_1ISI',
            'INS2015//12_14_20//INS_5PU_10PW_1ISI','INS2015//12_14_20//INS_5PU_10PW_5ISI']               #List of data to sort through
NPulse = [5,5,5,5,5,5,5,5,5,5,50,5,10,10,10,10,10,10,20,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
          5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,1,1,1,1,5,10,5,5,5,5,10,5,5,5,10,1,1,1,1,1,10,
          10,10,10,10,5,5,5,5,5,1,5,5,10,10,10,5,5,5,5,5,5,1,1,1,1,1,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
PWs = [0.5,10,0.2,0.5,0.7,1,5,5,10,10,0.5,10,0.2,0.2,0.5,0.7,1,5,0.2,0.7,10,10,0.2,0.2,0.5,0.5,1,1,0.2,5,0.1,0.2,0.5,1,5,0.2,0.2,0.5,0.5,1,1,5,5,0.2,0.5,1,1,5,5,10,0.5,1,5,5,10,10,0.1,0.2,1,10,1,1,1,1,0.2,0.5,
       1,1,5,5,0.2,0.5,1,5,0.2,0.2,0.5,0.5,1,1,5,5,0.2,0.5,1,5,5,10,0.1,0.2,0.5,1,1,5,10,1,0.5,5,5,1,1,5,5,10,0.5,0.5,5,10,10,0.5,0.2,0.5,5,10,10,0.2,0.2,0.5,1,10,1,0.2,0.5,1,1,0.2,0.5,1,5,10,0.2,
       0.5,1,5,10,1,5,10,10,10,10,5,10,0.5,1,1,1,5,10,10,10,10,0.2,0.5,1,5,10,10,0.1,0.1,0.2,0.5,0.8,1,5,10,1,1,5,5,5,10,10,0.5,5,10,10]
ISIs = [1,5,0.1,0.2,0.5,0.5,5,100,5,100,0.2,50,0.5,1,1,0.5,1,5,0.5,0.5,50,50,1,5,1,5,1,5,1,5,5,5,5,5,5,10,100,10,100,10,100,10,100,5,5,5,100,5,100,5,20,1,1,50,50,100,5,5,1,50,1,5,10,50,5,5,
        1,5,1,5,100,100,100,100,5,10,5,10,5,10,5,10,1,1,1,1,5,50,5,5,20,5,20,5,50,5,5,5,100,1,5,1,5,50,0.3,1,1,25,50,0.25,1,1,50,50,50,1,0.2,0.5,5,50,1,0.5,0.5,0.5,1,1,1,1,1,1,0.5,
        1,5,5,10,1,5,5,25,50,5,5,50,1,1,5,5,5,5,10,25,50,5,5,5,5,5,50,5,50,50,50,50,50,50,50,1,5,1,5,10,5,10,1,1,1,5]

AClass = np.zeros((len(PWs),))
for bc, word1 in enumerate(dataPath):
    if 'INS2102' in word1:
        AClass[bc] = 0
    elif 'INS2007' in word1:
        AClass[bc] = 1
    elif 'INS2008' in word1:
        AClass[bc] = 2
    elif 'INS2013' in word1:
        AClass[bc] = 3
    elif 'INS2015' in word1:
        AClass[bc] = 4
    else:
        print(str(word1)+' is Missed')

df = pd.DataFrame(columns=['DataID','EnergyPerPulse','ISI','NPulses','estCSD','xarray','yarray'])
eng = matlab.engine.start_matlab()          #Use the matlab backend for Info theory and Chaos calcs
SOS10 = sio.loadmat('SOSHighPass.mat')
SOS10 = np.ascontiguousarray(SOS10['SOS'])
#Place electrodes
x = 0.250*np.linspace(0,7,8)
y = 0.375*np.linspace(0,1,2)
ele_x, ele_y = np.meshgrid(x,y)
ele_pos = np.vstack((ele_x.flatten(), ele_y.flatten())).T
electrodeConfig = np.array([[10-1,12-1,14-1,16-1,9-1,11-1,13-1,15-1],[1-1,3-1,5-1,7-1,2-1,4-1,6-1,8-1]],np.int16)
counter = 0
for ck, word in enumerate(dataPath):
    stores = None             #Load all stores
    streamStore = 'streams'
    rawDataStore = 'TDT2'
    debug = 0
    stim = 0
    Type = 'LFP'
    SpksOrLFPs = [Type]
    PW = PWs[ck]
    ISI = ISIs[ck]
    NPul = NPulse[ck]
    if AClass[ck] == 0:
        power = np.array((-1.4, 37.2, 46.15, 58.6, 88, 94, 123, 182.62, 259, 313.6, 386.1, 414))
    elif AClass[ck] == 1:
        power = np.array((0,4,117,130,143.17,155.9,207,292,357,370,410,431))
    elif AClass[ck] == 2:
        power = np.array((0,4,117,130,143.17,155.9,207,292,357,370,410,431))
    elif AClass[ck] == 3:
        power = np.array((-1.1,62.1,77.42,87.4,101.2,115.9,130,184.34,257.3,308.8,360.7,374.4))
    elif AClass[ck] == 4:
        power = np.array((-1.1,62.1,77.42,87.4,101.2,115.9,130,184.34,257.3,308.8,360.7,374.4))
    try:
        SpikeClass = Spike_Processed(precurser+word,NPul,PW,ISI,power,stores,streamStore,debug,stim,SpksOrLFPs=SpksOrLFPs)

        #Spikes = SpikeClass.Spikes
        LFPs = SpikeClass.LFP

        epocedLFPs = SpikeClass.epocTrials(LFPs)
        sortedLFPs = SpikeClass.sortByStimCondition(epocedLFPs)
        waveClass = SpikeClass.sortByElectrode16(sortedLFPs)
        for energy in waveClass.keys():
            LFPArray = np.zeros((16,1526))
            curLFP = waveClass[energy]
            mLFP = np.mean(curLFP,axis=3)
           
            for ck in np.arange(0,7,1):
                for bc in np.arange(0,1,1):
                    curE = electrodeConfig[bc,ck]
                    LFPArray[curE] = mLFP[bc,ck,:]
            
            k = KCSD2D(ele_pos, LFPArray, h=1, sigma=1,xmin=0.0, xmax=1.75,ymin=0.0, ymax=0.375, ext_x=0.5,ext_y=0.5,n_src_init=1000, src_type='gauss', R_init=1.)
            k.cross_validate(Rs=np.linspace(0.01, 0.15, 15))
            est_csd = k.values('CSD')
            df.loc[-1] = [word,float(energy),ISI,NPul,est_csd,k.estm_x,k.estm_y]
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
    except Exception as error:
        # handle the exception
        print("An exception occurred:", type(error).__name__, "–", error) # An exception occurred: ZeroDivisionError – division by zero
        print('Brandon, check'+' '+word)
    if ck%50 == 0:
        savePW = saveKW+str(counter)+'.pkl'
        counter = counter + 1
        df.to_pickle(precurser+savePW)
        del df
        df = pd.DataFrame(columns=['DataID','EnergyPerPulse','ISI','NPulses','estCSD','xarray','yarray'])
#df.to_pickle('LFPCSD.pkl')
counter = counter+1
savePW = saveKW+str(counter)+'.pkl'
df.to_pickle(precurser+savePW)
pdb.set_trace()