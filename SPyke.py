#--------------------------------------------------------------------------------------------------------------------------------------------
# Authors: Brandon S Coventry            Wisconsin Institute for Translational Neuroengineering
# Date: 02/03/2023                       The night it was so cold, my car battery died at work... Wisconsin is cold
# Purpose: This is the header file for SPyke, a Python toolbox for analyzing Neural Spiking Data.
# Interface: At the moment, this uses TDT based reading and analysis, but can be extended to other systems.
# Revision History: Will be tracked in Github.
# Notes: N/A
#--------------------------------------------------------------------------------------------------------------------------------------------
# Imports go here. Global use imports will be put here, we will upload specifics as needed.
import numpy as np
import tdt              #For reading in tdt files
import matplotlib.pyplot as plt
import dask
import os
import glob
import sys
import pdb
import cupy as cp
from scipy.io import loadmat
from scipy.signal import sosfiltfilt
from scipy.signal import spectrogram
from scipy.signal import decimate
import fcwt         #For T-F decomposition https://github.com/fastlib/fCWT
import pickle as pkl
import pandas as pd
import PDEparams as pde
from scipy.optimize import curve_fit
class Spike(object):
    """
    Purpose: This is the main spike analysis method. Within will hold analysis operations, data loaders, and plotters.
    Class Methods: 
        __init__: Initialize the class, load class variables
            Inputs: data - This is a string containing the name of the datafile to be analyzed
                    *args - Arguments to be passed to TsData parent class.
                    stores - Names of stores to load from TDT tank. If empty, load all stores. Stores should be in form of list i.e. ['EPOCHS','Streams']
                    rawDataStore - Names of raw data variable. Should point to 
                    rz_sample_rate : int sample rate of RZ processor in TDT chain
                    si_sample_rate : int sample rate of SI processor in TDT chain
                    hasStim: Set to 0 if no stim present, set to 1 if stim is present
        loadData: This loads data into memory. 
        TODO: Check if GPU flag is needed with cp.get_array_module
    """
    def __init__(self, data, stores=None, streamStore = None, rawDataStore=None,debug=0, hasStim=0, rz_sample_rate=None, si_sample_rate=None, sample_delay=None,GPU=True,stimulation=True,SpksOrLFPs=['Spike','LFP'],**kwargs):
        super().__init__()
        self.stores = stores
        if stores==None:
            self.data = tdt.read_block(data)
        else:
            self.data = tdt.read_block(data,'store',self.stores)
        print(self.data)
        self.GPU = GPU
        self.storage = getattr(self.data,streamStore)
        self.raw = getattr(self.storage,rawDataStore)
        self.rz_sample_rate = rz_sample_rate
        self.si_sample_rate = si_sample_rate
        self.sample_delay = sample_delay
        self.fs = self.raw['fs']
        #if self.GPU == True:
            #self.rawData = cp.asarray(self.raw['data'])            #Get this into a CuPy for GPU processing
        #else:
            #self.rawData = dask.delayed(self.raw['data'])          #Get this into a dask array for quick processing.
        self.rawData = self.raw['data']
        self.channel = self.raw['channel']
        self.numChannel = len(self.channel)
        self.totSamp = len(self.rawData[0,:])
        self.ts = np.arange(0,self.totSamp/self.fs,1/self.fs)
        
        if debug == 0:
            self.lineHarmonicFilter()
            if len(SpksOrLFPs) > 1:
                self.filterData('Spike')
                self.filterData('LFP')
                self.filterData('Raw')
            elif SpksOrLFPs == 'Spike':
                self.filterData('Spike')
            elif SpksOrLFPs == 'LFP':
                self.filterData('LFP')
            else:
                raise TypeError('SpkOrLFP flag must be "Spike","LFP", or ["Spike","LFP"]')
        elif debug == 1:
            print('Time to Debug!')
            with open('Spike.pkl','rb') as f:
                self.Spikes = pkl.load(f)
            f.close()
            with open('LFP.pkl','rb') as f:
                self.LFP = pkl.load(f)
            f.close()
        
        # if 'Spike' in SpksOrLFPs:
        #     self.convertSpikes2Bin()
        #     self.sortSpikesKilosort()
        if hasStim == 1:
            self.extractStimEvents()
        #pdb.set_trace()
        
        
    
    def gpu2cpu(self,data2transfer):
        """
        This is a helper function to transfer gpu to cpu 
        Inputs: data2transfer - the data to move from GPU to CPU
        """
        return cp.asnumpy(data2transfer)
    
    def filterData(self,Type,channels=[]):
        """
        This function will filter the data for Spikes or for LFPs from predefined filters.
        Type - 'Spike' Will load a Chebychev type II filter with passband low = 500 and passband high = 5000
             - 'LFP' Will load a Chebychev type II filter with passband low = 3 and passband high = 500
        Channels - List of channels to filter. If empty, do all
        TODO - Independent channel filtering
             - Add filter coefficients and if statements for sampling rates at 50kHz. Right now filters are for 24415 Hz.
        """
        if Type == 'Spike':
            SOSSpike = loadmat('SOS_Spike')
            SOS = np.ascontiguousarray(SOSSpike['SOS_Spike'])   #For reasons unknown to me, matlab saves SOS coeffs as non C-contiguous arrays. Fixed here
            filteredData = sosfiltfilt(SOS,self.lineFilterRawData)
            self.Spikes = filteredData
        elif Type == 'LFP':
            SOSLFP = loadmat('SOS_LFP')
            SOS = np.ascontiguousarray(SOSLFP['SOS_LFP'])
            filteredData = sosfiltfilt(SOS,self.lineFilterRawData)
            self.LFP = filteredData
        elif Type == 'Raw':
            if self.fs < 25000:
                SOSRaw = loadmat('SOS_Raw')
                SOS = np.ascontiguousarray(SOSRaw['SOS'])
            elif self.fs > 25000:
                SOSRaw = loadmat('SOS_Raw_50')
                SOS = np.ascontiguousarray(SOSRaw['SOS'])
            filteredData = sosfiltfilt(SOS,self.lineFilterRawData)
            self.filteredRaw = filteredData
            "This filter is for removing DC offset in the raw signal"

        else:
            raise TypeError('Type must be "Spike" or "LFP"')
        return filteredData
        
    def lineHarmonicFilter(self):
       
        SOS60 = loadmat('SOS_60')
        SOS120 = loadmat('SOS_120')
        SOS240 = loadmat('SOS_240')
        SOS = np.ascontiguousarray(SOS60['SOS_60'])   #For reasons unknown to me, matlab saves SOS coeffs as non C-contiguous arrays. Fixed here
        filteredData1 = sosfiltfilt(SOS,self.rawData)
        SOS = np.ascontiguousarray(SOS120['SOS_120'])
        filteredData2 = sosfiltfilt(SOS,filteredData1)
        SOS = np.ascontiguousarray(SOS240['SOS_240'])
        filteredData3 = sosfiltfilt(SOS,filteredData2)
        self.lineFilterRawData = filteredData3
        
        return filteredData3
    
    def extractStimEvents(self,window = [-10,20]):
        """
        This function reads in stimulation times, calculates the number of unique events, and extracts signals around those times with a window specified by window
        Inputs - Window - Times around which to grab data. Negative values indicate times 
        TODO - Vectorize finding where each stim class falls during the recording period
        """
        self.stimTimes = self.data.scalars.semp.ts              #Get stim times from stores
        self.stimEvents = self.data.scalars.semp.data[0:3,:]    #We only need first 3 rows which carries stim period, num pulses, and pulse amplitude
        [stimClasses,uWhere] = np.unique(self.stimEvents,return_index=True,axis=1)        #Get the unique stim events. This might be 130Hz, -50ma, 30Hz, -70mA, etc.
        self.stimClass = np.sort(stimClasses,axis=0)                          #Order them just to make next data processing steps easier.
        nrowcol = np.shape(self.stimClass)                      #Helper variable to get the total number of stim classes
        self.numStims = nrowcol[1]
        #numRowsCols = np.shape(self.StimEvents)
        #self.numStims = numRowsCols[1]
        stimWhere = []
        for ck in range(self.numStims):                         #Hugely inefficient. Will vectorize. Till then, loop through the stim times and find where each stim class is.
            curStim = self.stimClass[:,ck]
            curStimRe = np.asarray([curStim[1],curStim[2],curStim[0]])
            idxStore = []
            for bc in range(len(self.stimTimes)):
                if np.array_equal(curStimRe,self.stimEvents[:,bc]):
                    idxStore.append(bc)
            stimWhere.append(idxStore)
            #stimWhere[ck,:] = np.where(self.stimEvents==curStim)
        self.stimWhere = stimWhere
        #Now get loop through and grab stim windows around the applied window
        try:
            windowSize = np.abs(window[0])+np.abs(window[1])
        except:
            raise ValueError('window should be a list or array of two values. window[0] = low time around which to grab data. window[1] = high time around which to grab data')
        windowSamples = len(np.arange(0,windowSize,1/self.fs))         #Again potentially silly way to do this, but works for now. Get the total number of samples needed per win
        [numStimClasses,self.numTrials] = np.shape(stimWhere)
        self.SpikeStore = dict()     #Convinient Storage! Yay!
        self.LFPStore = dict()
        self.RawStore = dict()
        for ck in range(numStimClasses):
            trialStoreArraySpikes = np.empty((self.numTrials,self.numChannel,windowSamples))
            trialStoreArrayLFPs = np.empty((self.numTrials,self.numChannel,windowSamples))
            trialStoreArrayRaw = np.empty((self.numTrials,self.numChannel,windowSamples))
            trialStoreArraySpikes[:] = np.nan                               #Do Nans because we have to account for first stimulus that may be less than window size.
            trialStoreArrayLFPs[:] = np.nan
            trialStoreArrayRaw[:] = np.nan
            storeKey = str(self.stimClass[0,ck])+'_'+str(self.stimClass[1,ck])+'_'+str(self.stimClass[2,ck])
            for bc in range(self.numTrials):
                curTimeIDX = self.stimWhere[ck][bc]
                curTime = self.stimTimes[curTimeIDX]
                if curTime+window[0] < 0:                  #Since this is shorter than winlow, we will add Nans to beginning of store file so that we are aligned with other trials
                    winLow = 0
                    winHighIDX = window[1]*self.fs
                    stimOn = np.where(self.ts == curTime)
                    if np.any(stimOn) == False:
                        diffTs = np.abs(self.ts-curTime)
                        whIDX = np.where(diffTs == min(diffTs))
                        stimOn = whIDX[0][0]
                    winHigh = round(np.asscalar(stimOn + winHighIDX))
                    shortDataSpikes = self.Spikes[:,winLow:winHigh]
                    shortDataLFP = self.LFP[:,winLow:winHigh]
                    shortDataRaw = self.rawData[:,winLow:winHigh]
                    [_,totSam] = np.shape(shortDataSpikes) 
                    numSamplesMissing = windowSamples-totSam
                    nanArraySpike = np.empty((self.numChannel,numSamplesMissing))
                    nanArraySpike[:] = np.nan
                    nanArrayLFP = np.empty((self.numChannel,numSamplesMissing))
                    nanArrayLFP[:] = np.nan
                    nanArrayRaw = np.empty((self.numChannel,numSamplesMissing))
                    nanArrayRaw[:] = np.nan
                    catArraySpike = np.empty((self.numChannel,windowSamples))
                    catArraySpike[:,0:numSamplesMissing] = nanArraySpike
                    catArraySpike[:,numSamplesMissing:numSamplesMissing+totSam] = shortDataSpikes
                    trialStoreArraySpikes[bc,:,:] = catArraySpike
                    catArrayLFP = np.empty((self.numChannel,windowSamples))
                    catArrayLFP[:,0:numSamplesMissing] = nanArrayLFP
                    catArrayLFP[:,numSamplesMissing:numSamplesMissing+totSam] = shortDataLFP
                    trialStoreArrayLFPs[bc,:,:] = catArrayLFP

                    catArrayRaw = np.empty((self.numChannel,windowSamples))
                    catArrayRaw[:,0:numSamplesMissing] = nanArrayRaw
                    catArrayRaw[:,numSamplesMissing:numSamplesMissing+totSam] = shortDataRaw
                    trialStoreArrayRaw[bc,:,:] = catArrayRaw
                else:
                    stimOn = np.where(self.ts == curTime)
                    if np.any(stimOn) == False:         #Sometimes precision is off. Use this to find correct stim index if stimOn is empty
                        diffTs = np.abs(self.ts-curTime)
                        whIDX = np.where(diffTs == min(diffTs))
                        stimOn = whIDX[0][0]
                    winLowIDX = np.abs(window[0])*self.fs
                    winHighIDX = window[1]*self.fs
                    try:
                        winLow = round(np.asscalar(stimOn - winLowIDX))
                        winHigh = round(np.asscalar(stimOn + winHighIDX))
                    except:
                        print('well that was weird')
                        pdb.set_trace()
                    winHighDiff = windowSamples - len(range(winLow,winHigh))
                    
                    winHigh = winHigh + winHighDiff           #Make sure arrays are same shape
                    trialStoreArraySpikes[bc,:,:] = self.Spikes[:,winLow:winHigh]          #Store all trials, all electrodes, windows into arrays that will be loaded into a dictionary
                    trialStoreArrayLFPs[bc,:,:] = self.LFP[:,winLow:winHigh]
                    trialStoreArrayRaw[bc,:,:] = self.rawData[:,winLow:winHigh]
            self.SpikeStore[storeKey] = trialStoreArraySpikes
            self.LFPStore[storeKey] = trialStoreArrayLFPs
            self.RawStore[storeKey] = trialStoreArrayRaw
    
    def sortSpikesKilosort(self,matlabKilosortPath = 'C://Users//coventry//CodeRepos//Kilosort-main//Kilosort-main'):
        """
        This function will create a call to matlab to run units for spike sorting. Note requires matlab python plugin to run.
        Reguires a compiled version of kilosort to work. Operates by opening gui for visualization of traces
        """
        import matlab.engine               #Get a hold of matlab
        eng = matlab.engine.start_matlab() #Start Matlab
        ksortPath = eng.genpath(r'C://Users//coventry//CodeRepos//Kilosort-main//Kilosort-main')    #Need to tell python where kilosort is
        eng.addpath(ksortPath,nargout=0)
        eng.kilosort()                  #Run kilosort
    
    def convertSpikes2Bin(self,scaleFactor=1e6,binName = 'SpikeKilosort.bin'):
        """
        This function converts Spike data into a .bin file for kilosort
        Inputs - scaleFactor: Data is typically float like, but kilosort needs int16. To get the resolution needed for uV, 
        need to scale data by this factor so that every data point is not immediately cast to 0 from conversion from float to int 16.
        This varies based on recording setup, electrodes. For example, TDT recommends 1e6 conversion for their systems and converts to
        microvolts.
        Bin file is a matrix of shape [nelectrodes x ntimepoints]. Our class varialbe Spike is already in this form.
        """
        try:
            data = (self.Spikes)*scaleFactor
        except:
            raise RuntimeError('Spikes not found in class. Can only run this conversion if Spikes are available')
        with open(binName, 'wb') as fh:
            pkl.dump(data, fh)
    
    def plotSampleWaveform(self,data,channel,time2Plot=[]):
        """
        This is a sample helper function just to plot some data.
        """
        for ck in range(len(channel)):
            plt.figure(ck)
            if bool(time2Plot)==False:
                plt.plot(self.ts,data[channel[ck],:])
            else:
                plt.plot(self.ts[time2Plot],data[channel[ck],time2Plot])
            plt.show()
    
    def stimArtifactRemoval(self,algo='ERAASER'):
        """
        Call matlab to run the Eraaser algorithm (https://iopscience.iop.org/article/10.1088/1741-2552/aaa365) on raw traces. Requires 
        That input data be on stimulus aligned trial data (that we can get from the extractStimEvents class function)
        Inputs: - algo - Only ERAASER is available at the moment. Add more as I find them
        """
        if algo == 'ERAASER':
            import matlab.engine               #Get a hold of matlab
            self.cleanData = dict()
            eng = matlab.engine.start_matlab() #Start Matlab
            keyList = self.RawStore.keys()
            keyListLen = len(keyList)
            ksortPath = eng.genpath(r'C://Users//coventry//CodeRepos//eraaser-master')    #Need to tell python where kilosort is
            eng.addpath(ksortPath,nargout=0)
            for key in self.RawStore:
                curData = self.RawStore[key]
                curDataErase = np.ndarray.tolist(self.getTensor(curData))
                self.cleanData[key] = eng.SPykeEraaser(curDataErase)
                pdb.set_trace()
        if algo == 'Template':
            """
            The Template command looks for pattern-like activity and compares to stimulation atributes.
            """
            self.pulseTime = 0.000001             #Pulse time in seconds
            self.pulseFreq = 130                  #Pulse Frequnecy in Hz
            import pywt
            self.dwt = pywt.dwt(self.filteredRaw, 'haar', mode='symmetric', axis=-1)




    
    def getTensor(self,data):
        """
        This helper function converts data dictionary to a data tensor suitable for ERAASER. 
        """
        return np.transpose(data,[0,2,1])

    def getLFPSpectrogram(self,channels,plotFlag=True):
        [numChannels,Siglen] = np.shape(self.LFP)
        Sxx = []
        t = []
        f = []
        for ck in range(len(channels)):
            curF,curT,curSxx = spectrogram(self.LFP[ck,:],self.fs)
            Sxx.append(curSxx)
            t.append(curT)
            f.append(curF)
            if plotFlag == True:
                plt.pcolormesh(curT, curF, curSxx, shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()
        return Sxx,t,f
class Spike_Processed(object):
    """
    Purpose: This is the main spike analysis method. Within will hold analysis operations, data loaders, and plotters. For Brandon's INS Data
    Class Methods: 
        __init__: Initialize the class, load class variables
            Inputs: data - This is a string containing the name of the datafile to be analyzed
                    *args - Arguments to be passed to TsData parent class.
                    stores - Names of stores to load from TDT tank. If empty, load all stores. Stores should be in form of list i.e. ['EPOCHS','Streams']
                    rawDataStore - Names of raw data variable. Should point to 
                    rz_sample_rate : int sample rate of RZ processor in TDT chain
                    si_sample_rate : int sample rate of SI processor in TDT chain
                    hasStim: Set to 0 if no stim present, set to 1 if stim is present
        loadData: This loads data into memory. 
        TODO: Check if GPU flag is needed with cp.get_array_module
    """
    def __init__(self, data, numpulses, PW, ISI, power, stores=None, streamStore = None,debug=0, hasStim=0, rz_sample_rate=None, si_sample_rate=None, sample_delay=None,GPU=True,stimulation=True,SpksOrLFPs=['Spike','LFP'],**kwargs):
        super().__init__()
        
        if stores == 0:
            stores = None
        self.stores = stores
        self.numCores = 6         #For CWT, set to number of CPU cores
        if stores==None:
            self.data = tdt.read_block(data)
        else:
            self.data = tdt.read_block(data,'store',self.stores)
        print(self.data)
        self.numpulses = numpulses
        self.PW = PW
        self.ISI = ISI
        self.power = power
        self.GPU = GPU
        
        self.storage = getattr(self.data,streamStore)
        if len(SpksOrLFPs) > 1:
            self.Spikes = getattr(self.storage,'Spks')
            self.LFP = getattr(self.storage,'LFPs')
            self.fs = self.LFP['fs']
            self.channel = self.LFP['channel']
            self.numChannel = len(self.channel)
            self.LFP = self.LFP['data']
            self.Spikes = self.Spikes['data']
            self.totSamp = len(self.Spikes[0,:])
            self.ts = np.arange(0,self.totSamp/self.fs,1/self.fs)
            
        elif SpksOrLFPs[0] == 'Spike':
            self.Spikes = getattr(self.storage,'Spks')
            self.fs = self.Spikes['fs']
            self.channel = self.Spikes['channel']
            self.numChannel = len(self.channel)
            self.Spikes = self.Spikes['data']
            self.totSamp = len(self.Spikes[0,:])
            self.ts = np.arange(0,self.totSamp/self.fs,1/self.fs)
        elif SpksOrLFPs[0] == 'LFP':
            self.LFP = getattr(self.storage,'LFPs')
            self.fs = self.LFP['fs']
            self.channel = self.LFP['channel']
            self.numChannel = len(self.channel)
            self.LFP = self.LFP['data']
            self.LFP = self.lineHarmonicFilter(self.LFP)
            self.LFP = decimate(self.LFP,16)
            self.fs = self.fs/16
            self.totSamp = len(self.LFP[0,:])
            self.ts = np.arange(0,self.totSamp/self.fs,1/self.fs)
            #self.waveletDecomposition()
            tISI = float(ISI)*0.001
            fundFreq = 1/float(tISI)
            self.waveletDecompositionISI(fundFreq)
            
        self.epocs = self.data.epocs.RZ2T.onset
        self.getEpocSamp()
        if 'self.LFP' in locals():
            pass
        # self.alphaTrials = self.epocTrials(self.alpha)
        # self.betaTrials = self.epocTrials(self.beta)
        # self.thetaTrials = self.epocTrials(self.theta)
        # self.lowGammaTrials = self.epocTrials(self.lowgamma)
        # self.highGammaTrials = self.epocTrials(self.highgamma)
        #self.epochCWT = self.epocTrials(self.wavelet)
        #pdb.set_trace()
        self.ISITrials = self.epocTrials(self.ISILock)
        #self.epochCWT = self.epocTrials(self.wavelet)
        #pdb.set_trace()
        #self.epocLFP = self.epocTrials(self.LFP)
        self.epocLFP = self.epocTrials(self.LFP)
        
        self.readINSLaserVoltages()
        # self.epocedAlpha = self.sortByStimCondition(self.alphaTrials)
        # self.epochedBeta = self.sortByStimCondition(self.betaTrials)
        # self.epochedTheta = self.sortByStimCondition(self.thetaTrials)
        # self.epochedLowGamma = self.sortByStimCondition(self.lowGammaTrials)
        # self.epochedHighGamma = self.sortByStimCondition(self.highGammaTrials)
        #if self.GPU == True:
            #self.rawData = cp.asarray(self.raw['data'])            #Get this into a CuPy for GPU processing
        #else:
            #self.rawData = dask.delayed(self.raw['data'])          #Get this into a dask array for quick processing.
        
        self.epochedISI = self.sortByStimCondition(self.ISITrials)
        # if 'Spike' in SpksOrLFPs:
        #     self.convertSpikes2Bin()
        #     self.sortSpikesKilosort()
        if hasStim == 1:
            self.extractStimEvents()
        #pdb.set_trace()
        
        
    def getEpocSamp(self):
        # This helper function finds the start samples of stimulation epocs.
        
        lenEpoch = len(self.epocs)
        epochSamp = np.zeros(lenEpoch,)
        
        for ck in range(lenEpoch):
            
            epochSamp[ck] = int(np.argwhere(abs((self.ts-self.epocs[ck]))<.0003))
        self.epochSamp = epochSamp
       

    def gpu2cpu(self,data2transfer):
        """
        This is a helper function to transfer gpu to cpu 
        Inputs: data2transfer - the data to move from GPU to CPU
        """
        return cp.asnumpy(data2transfer)
    def readINSLaserVoltages(self):
        
        curVales = loadmat('INSvoltagevals.mat')
        curVales = curVales['stimpeaks']
        [rows,cols] = np.shape(curVales)
        voltageVals = np.zeros((rows,))
        for ck in range(rows):
            
            voltageVals[ck] = curVales[ck][0]
        self.voltageVals = voltageVals
        uniqueVals = np.unique(self.voltageVals)
        #Get energy per pulse
        self.energyPerPulse = (self.power*0.001)*(self.PW*0.001)*1000
        
        numUnique = len(uniqueVals)
        uniqueWhere = {}
        for bc in range(numUnique):
            curUniq = uniqueVals[bc]
            curEnergy = self.energyPerPulse[bc]
            uniWhere = np.argwhere(curUniq==self.voltageVals)
            uniqueWhere[str(curEnergy)] = uniWhere
        self.uniqueWhere = uniqueWhere

    def filterData(self,Data,channels=[]):
        """
        This function will filter the data for Spikes or for LFPs from predefined filters.
        Type - 'Spike' Will load a Chebychev type II filter with passband low = 500 and passband high = 5000
             - 'LFP' Will load a Chebychev type II filter with passband low = 3 and passband high = 500
        Channels - List of channels to filter. If empty, do all
        TODO - Independent channel filtering
             - Add filter coefficients and if statements for sampling rates at 50kHz. Right now filters are for 24415 Hz.
        """
    
        if self.fs < 25000:                   #For reasons unknown to me, matlab saves SOS coeffs as non C-contiguous arrays. Fixed here
            SOSRaw = loadmat('SOS_Raw')
            SOS = np.ascontiguousarray(SOSRaw['SOS'])
        elif self.fs > 25000:
            SOSRaw = loadmat('SOS_Raw_50')
            SOS = np.ascontiguousarray(SOSRaw['SOS'])
        filteredData = sosfiltfilt(SOS,Data)
        return filteredData
        #"This filter is for removing DC offset in the raw signal"
        
    def lineHarmonicFilter(self,DATA):
       
        SOS60 = loadmat('SOS_60')
        SOS120 = loadmat('SOS_120')
        SOS240 = loadmat('SOS_240')
        SOS = np.ascontiguousarray(SOS60['SOS_60'])   #For reasons unknown to me, matlab saves SOS coeffs as non C-contiguous arrays. Fixed here
        filteredData1 = sosfiltfilt(SOS,DATA)
        SOS = np.ascontiguousarray(SOS120['SOS_120'])
        filteredData2 = sosfiltfilt(SOS,filteredData1)
        SOS = np.ascontiguousarray(SOS240['SOS_240'])
        filteredData3 = sosfiltfilt(SOS,filteredData2)
        self.lineFilterRawData = filteredData3
        
        return filteredData3
    def chaosFilter(self,data):
        SOS = loadmat('chaosFilt')
        SOS = np.ascontiguousarray(SOS['SOS'])
        filteredData1 = sosfiltfilt(SOS,data)
        return filteredData1
    def waveletDecomposition(self,flow=3,fhi=200,fn=250):
        #This helper function computes the CWT using the fCWT method.
        #This helper function computes the CWT using the fCWT method.
        theta = [4,8]
        alpha = [8,13]
        beta  = [13,30]
        lowgamma = [30,80]
        highgamma = [80,200]
        import scipy.integrate as tegral
        self.wavelet = np.zeros((self.numChannel,fn,self.totSamp))
        self.theta = np.zeros((self.numChannel,self.totSamp))
        self.alpha = np.zeros((self.numChannel,self.totSamp))
        self.beta = np.zeros((self.numChannel,self.totSamp))
        self.lowgamma = np.zeros((self.numChannel,self.totSamp))
        self.highgamma = np.zeros((self.numChannel,self.totSamp))
        
        for ck in range(4):
            freqs, cwt = fcwt.cwt(self.LFP[ck,:], int(np.floor(self.fs)), flow, fhi, fn, nthreads=self.numCores)
            #alphaWhere = np.where(np.logical_and(freqs>=alpha[0], freqs<=alpha[1]))
            #thetaWhere = np.where(np.logical_and(freqs>=theta[0], freqs<=theta[1]))
            #betaWhere = np.where(np.logical_and(freqs>=beta[0], freqs<=beta[1]))
            #lowGammaWhere = np.where(np.logical_and(freqs>=lowgamma[0], freqs<=lowgamma[1]))
            #highGammaWhere = np.where(np.logical_and(freqs>=highgamma[0], freqs<=highgamma[1]))
            wavelet = np.square(np.abs(cwt))
            #pdb.set_trace()
            #self.theta[ck,:] = np.squeeze(2*tegral.simpson(wavelet[thetaWhere,:],axis=1))
            #self.alpha[ck,:] = np.squeeze(2*tegral.simpson(wavelet[alphaWhere,:],axis=1))
            #self.beta[ck,:] = np.squeeze(2*tegral.simpson(wavelet[betaWhere,:],axis=1))
            #self.lowgamma[ck,:] = np.squeeze(2*tegral.simpson(wavelet[lowGammaWhere,:],axis=1))
            #self.highgamma[ck,:] = np.squeeze(2*tegral.simpson(wavelet[highGammaWhere,:],axis=1))
            self.wavelet[ck,:,:] = np.abs(cwt)
            #pdb.set_trace()
        self.freqs = freqs
    def waveletDecompositionISI(self,ISI,flow=10,fhi=5010,fn=700):
        #This helper function computes the CWT using the fCWT method.
        #This helper function computes the CWT using the fCWT method.
        
        
        self.wavelet = np.zeros((self.numChannel,fn,self.totSamp))
        self.ISILock = np.zeros((self.numChannel,self.totSamp))
        
        for ck in range(16):
            freqs, cwt = fcwt.cwt(self.LFP[ck,:], int(np.floor(self.fs)), flow, fhi, fn, nthreads=self.numCores)
            index = (np.abs(freqs - ISI)).argmin()
            wavelet = np.square(np.abs(cwt))
            self.ISILock[ck,:] = np.squeeze(wavelet[index,:])
            
            #self.theta[ck,:] = np.squeeze(2*tegral.simpson(wavelet[thetaWhere,:],axis=1))
            #self.alpha[ck,:] = np.squeeze(2*tegral.simpson(wavelet[alphaWhere,:],axis=1))
            #self.beta[ck,:] = np.squeeze(2*tegral.simpson(wavelet[betaWhere,:],axis=1))
            #self.lowgamma[ck,:] = np.squeeze(2*tegral.simpson(wavelet[lowGammaWhere,:],axis=1))
            #self.highgamma[ck,:] = np.squeeze(2*tegral.simpson(wavelet[highGammaWhere,:],axis=1))
            #self.wavelet[ck,:,:] = np.abs(cwt)
            
        self.freqs = freqs

    def epocTrials(self,data):
        
        
        lenTrials = self.totSamp
        lenEpocs = len(self.epocs)
        LFP_Trials = {}
        
        for bc in range(self.numChannel):
            
            curData = data[bc,:]
            dataLen = len(curData[int(self.epochSamp[0]):int(self.epochSamp[1])])
            lenTrials = dataLen
            epocedData = np.zeros((lenEpocs,lenTrials))
            for ck in range(lenEpocs-1):
                    newRangeStart = int(self.epochSamp[ck])
                    newRangeEnd = int(self.epochSamp[ck+1])
                    if newRangeEnd-newRangeStart == 1525:
                        newRangeEnd = newRangeEnd+1
                    try:
                        epocedData[ck,:] = curData[newRangeStart:newRangeEnd]
                    except:
                        pdb.set_trace()
            LFP_Trials[str(bc)] = epocedData
        
        return LFP_Trials

    def sortByStimCondition(self,data):
        numElectrodes = len(data.keys())
        epochedLFPs = {}
        
        for ck in range(numElectrodes):
            curData = data[str(ck)]
            stimCondition = {}
            for bc in range(len(self.uniqueWhere.keys())):
                curStimValKey = list(self.uniqueWhere.keys())[bc]
                curStimVal = np.squeeze(self.uniqueWhere[curStimValKey])
                stimCondition[str(curStimValKey)] = curData[curStimVal,:]
            epochedLFPs[str(ck)] = stimCondition
            #pdb.set_trace()
        return epochedLFPs  
    
    def getMeanSdEr(self,data):
        electrodes = len(data.keys())
        dataMean = {}
        dataSdEr = {}
        for ck in range(electrodes):
            curData = data[str(ck)]
            dataMeanEnergy = {}
            dataSdErEnergy = {}
            for key in curData.keys():
                curDataE = curData[key]
                [r,c] = np.shape(curDataE)
                meanData = np.mean(curDataE,axis=0)
                sddvData = np.std(curDataE,axis=0)
                sderData = sddvData/np.sqrt(r)
                dataMeanEnergy[key] = meanData
                dataSdErEnergy[key] = sderData
            dataMean[str(ck)] = dataMeanEnergy
            dataSdEr[str(ck)] = dataSdErEnergy
        return dataMean,dataSdEr

    def sortMeanByElectrode16(self,meanData,numSamples=1526):
        
        #The -1 on each of the entries below is just to help me convert between 0 indexing (Python) and 1 indexing (electrode array)
        self.electrodeConfig = np.array([[10-1,12-1,14-1,16-1,9-1,11-1,13-1,15-1],[1-1,3-1,5-1,7-1,2-1,4-1,6-1,8-1]],np.int16)
        sortedMean = {str(key): {} for key in self.energyPerPulse}
        electrodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        
        for ck in range(len(electrodes)):
            curData = meanData[str(ck)]
            for key in curData.keys():
                currentData = curData[key]
                sortedMean[key][str(ck)] = currentData
        sortedE = {}
        for key in sortedMean.keys():
            cDat = sortedMean[key]
            electrodeArray = np.zeros((2,8,numSamples))
            for newkey in cDat.keys():
                curE = np.where(int(newkey)==self.electrodeConfig)
                electrodeArray[curE[0][0],curE[1][0],:]=cDat[newkey]
            sortedE[key] = electrodeArray
        
        return sortedE
    
    def sortByElectrode16(self,meanData,numSamples=1526):
        
        #The -1 on each of the entries below is just to help me convert between 0 indexing (Python) and 1 indexing (electrode array)
        self.electrodeConfig = np.array([[10-1,12-1,14-1,16-1,9-1,11-1,13-1,15-1],[1-1,3-1,5-1,7-1,2-1,4-1,6-1,8-1]],np.int16)
        sortedMean = {str(key): {} for key in self.energyPerPulse}
        electrodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        
        for ck in range(len(electrodes)):
            curData = meanData[str(ck)]
            for key in curData.keys():
                currentData = curData[key]
                sortedMean[key][str(ck)] = currentData
        sortedE = {}
        
        for key in sortedMean.keys():
            cDat = sortedMean[key]
            test2GetTrials = cDat['0']
            [nTrials,nt] = np.shape(test2GetTrials)
            electrodeArray = np.zeros((2,8,numSamples,nTrials))
            for newkey in cDat.keys():
                curE = np.where(int(newkey)==self.electrodeConfig)
                electrodeArray[curE[0][0],curE[1][0],:,:]=np.transpose(cDat[newkey])
            sortedE[key] = electrodeArray
        
        return sortedE
    
    def sortByElectrode16_MonoEnergy(self,meanData,numSamples=1526):
        self.electrodeConfig = np.array([[10-1,12-1,14-1,16-1,9-1,11-1,13-1,15-1],[1-1,3-1,5-1,7-1,2-1,4-1,6-1,8-1]],np.int16)
        sortedMean = {str(key): {} for key in self.energyPerPulse}
        electrodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        
        nTrials = 200
        electrodeArray = np.zeros((2,8,numSamples,nTrials))
        for newkey in meanData.keys():
            cDat = meanData[newkey]
            curE = np.where(int(newkey)==self.electrodeConfig)
            electrodeArray[curE[0][0],curE[1][0],:,:]=np.transpose(cDat)
        sortedE = electrodeArray
        return sortedE

    def convert2DF(self,data):
        # Assumes the DF convention of PDE params, ie
        """
        t     x      y       LFP(t,x,y)
        """
        dfDictionary = {str(key): {} for key in self.energyPerPulse}
        
        for key in data.keys():
            curData = data[key]
            dataSize = np.shape(curData)
            dataList = np.zeros((dataSize[0]*dataSize[1]*dataSize[2],4))
            counter = 0
            ts = np.arange(0,1,1/dataSize[2])
            for ck in range(dataSize[2]):
                for jt in range(dataSize[1]):
                    for bc in range(dataSize[0]):
                        dataList[counter,0] = ts[ck]
                        dataList[counter,2] = bc*0.500
                        dataList[counter,1] = jt*0.250
                        dataList[counter,3] = curData[bc,jt,ck]
                        counter = counter + 1
            curDF = pd.DataFrame(dataList, columns =['t', 'x', 'y', 'f']) 
            dfDictionary[key] = curDF
        return dfDictionary
    def convert2DFCOG(self,data,COGX,COGY):
        # Assumes the DF convention of PDE params, ie
        """
        t     x      y       LFP(t,x,y)
        """
        [ny,nx,nt] = np.shape(data)

    def WaveEq(self,z, t, grid, c, b):
        
        '''The input z corresponds to the current state of the system, and it's a flattened vector in both the number
        of outputs and the spatial dimensions. 
    
        t is the current time.
    
        grid is the spatial grid of the model.

        alpha and gamma correspond to the unknown parameters.
        '''
        '''Here we obtain both functions by reshaping the input: we divide it into 1 portions (number of outputs)
    of whatever shape the spatial grid has (this is what the -1 stands for)'''
    
        u = z.reshape(-1)
        
        '''Now we reshape both functions using the actual dimensions of the spatial grid.
        
        The shape of the spatial grid array is given by:
        
                            (number of points in dim 1, number of points in dim 2, ..., ndims)
                            
        with ndims the number of spatial dimensions. This is due to the fact that we have a grid of 
                            
                            (number of points in dim 1)x(number of points in dim 2)
                            
        elements, but each element is ndims-dimensional. Therefore, we must take the first "ndims" elements of the
        shape of the grid, ignoring the last one, to reconstruct the shape of each function. The slice with [:-1]
        stands for "all elements up to and excluding the last one".
        '''
        
        u = u.reshape(grid.shape[:-1])
        
        
        '''The grid has the form
                    
                    (x0,y0), (x0,y1), ...
                    (x1,y0), (x1,y1), ...
                    
        So that:
            x0 = first element of the first element of the first row of the grid: grid[0,0,0]
            x1 = first element of the first element of the second row of the grid: grid[1,0,0]
            y0 = second element of the first element of the first row of the grid: grid[0,0,1]
            y1 = second element of the second element of the first row of the grid: grid[0,1,1]
        '''
        
        dx = grid[1,0,0]-grid[0,0,0]
        dy = grid[0,1,1]-grid[0,0,1]
        dt = 1/1526           #fs period
        # We initialise the spatial derivatives we need as empty arrays of the same shape of our N and L functions
        
        dudx = np.gradient(u, axis=0)/dx
        dudy = np.gradient(u, axis=1)/dy
        
         # Second-order derivatives
    
        dudxx = np.gradient(dudx, axis=0)/dx
        dudyy = np.gradient(dudy, axis=1)/dy
        
        # np.gradient(array, axis) returns centred 1st-order differences between values along x (rows, axis=0)
        # or y (columns, axis=1), plus forward/backward differences for the left/right boundaries
        #Now get the time derivatives

        dudt = (np.power(c,2)*t*(dudxx+dudyy))-b*u
        
        #Set up boundary conditions
        dudt[0,:] = c*dudx[0,:]
        dudt[:,0] = c*dudy[:,0]
        dudt[-1,:] = -c*dudx[0,:]
        dudt[:,-1] = -c*dudy[:,0]
        
        return dudt.reshape(-1)
    
    def initValWaveEq(self,LFPinit=0):
        return np.zeros((1,))
    
    def runWaveFit(self,LFPdf,cBound,bBound):
        
        my_model = pde.PDEmodel(LFPdf, self.WaveEq, [self.initValWaveEq], 
                        bounds=[cBound, bBound], param_names=[r'$\velocity$', r'$\beta$'], 
                        nvars=1, ndims=2, nreplicates=0, obsidx=None, outfunc=None)
        my_model.fit()
        my_model.best_params
        my_model.best_error
        print('Best Error = ' + str(my_model.best_error))
        return [my_model.best_params,my_model.best_error]
    
    def COG(self,LFP):
        xvals = [0, 0.250, 0.500, 0.750, 1.00, 1.25, 1.50, 1.75]
        yvals = [0, 0.375]
        COGLFP = {}
        for key in LFP.keys():
            curDat = LFP[key]
            [nx,ny,nz] = np.shape(curDat)
            massTS = np.zeros((2,nz))
            for ck in range(nz):
                xLFP = []
                yLFP = []
                sLFP = []
                curtime = np.squeeze(curDat[:,:,ck])
                
                for bc in range(ny):
                    for jk in range(nx):
                        curX = xvals[bc]
                        curY = yvals[jk]
                        xLFP.append(curX*curtime[jk,bc])
                        yLFP.append(curY*curtime[jk,bc])
                        sLFP.append(curtime[jk,bc])
                massTS[0,ck] = (np.sum(xLFP))/np.sum(sLFP)
                massTS[1,ck] = (np.sum(yLFP))/np.sum(sLFP)
            COGLFP[key] = massTS
        return COGLFP

    def trackLFP(self,meanArray):
        #Mean array is of size 2x8x1526
        
        [y,x,ts] = np.shape(meanArray)
        maxXArray = np.zeros((ts,))
        maxYArray = np.zeros((ts,))
        minXArray = np.zeros((ts,))
        minYArray = np.zeros((ts,))
        for ck in range(ts):
            curData=np.squeeze(meanArray[:,:,ck])
            maxCurData = np.max(curData)
            minCurData = np.min(curData)
            [row,col] = np.where(curData==maxCurData)
            maxXArray[ck] = col
            maxYArray[ck] = row
            [row,col] = np.where(curData==minCurData)
            minXArray[ck] = col
            minYArray[ck] = row
        return [maxXArray,maxYArray,minXArray,minYArray]

    def cogMean(self,data):
        from scipy.ndimage import center_of_mass
        [ny,nx,nt] = np.shape(data)
        COGX = np.zeros((nt,))
        COGY = np.zeros((nt,))
        for ck in range(nt):
            
            curLFP = data[:,:,ck]
            [yp,xp] = center_of_mass(curLFP)
            COGX[ck] = xp*0.250
            COGY[ck] = yp*0.375
        return COGX,COGY

    def calcDistVelocity(self,x,y):
        
        nt = np.shape(x)
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        step_size = np.sqrt(dx**2+dy**2)
        instV = step_size/1526*1000
        accel = np.diff(instV)
        accelSDDV = 5*np.std(accel[0:305])
        
        accelWhere = np.where(accel > accelSDDV)
        accelWhere = accelWhere[0]
        if len(accelWhere) > 1:
            waveTime = (accelWhere[-1] - accelWhere[0])/1526*1000
        elif len(accelWhere) == 1:
            waveTime = 1/1526*1000
        else:
            waveTime = -1
        cumulative_distance = np.concatenate(([0], np.cumsum(step_size)))
        return step_size,instV,waveTime


    
    def normalizeMeanLFP(self,data):
        max = -100000
        min = 1000000
        [ny,nx,nt] = np.shape(data)
        normLFP = np.zeros((ny,nx,nt))
        for ck in range(ny):
            for bc in range(nx):
                curLFP = np.abs(data[ck,bc,:])
                maxLFP = np.max(curLFP)
                minLFP = np.min(curLFP)
                if maxLFP>=max:
                    max = maxLFP
                if minLFP <= min:
                    min = minLFP
        for ck in range(ny):
            for bc in range(nx):
                curLFP = data[ck,bc,:]
                normLFP[ck,bc,:] = self.minMaxNorm(curLFP,min,max)
        return normLFP
    def normalizeByLFP(self,data):
        [ny,nx,nt] = np.shape(data)
        normLFP = np.zeros((ny,nx,nt))
        for ck in range(ny):
            for bc in range(nx):
                curLFP = data[ck,bc,:]
                normLFP[ck,bc,:] = self.minMaxNorm(curLFP,np.min(curLFP),np.max(curLFP))
        return normLFP
    def minMaxNorm(self,data,min,max,a=-1,b=1):
        return (((data-min)/max-min))
    
    def plotCOG(self,COGX,COGY):
        tsamp = np.shape(COGX)
        ts = np.arange(0,1,1/tsamp[0])
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(COGX,COGY,ts)
        plt.show()
        
    
    def convert2Array(self,data):
        energyArray = np.zeros((2,8,1526,12))
        energy = data.keys()
        counter = 0
        for keys in data.keys():
            energyArray[:,:,:,counter] = data[keys]
            counter = counter + 1 
        return energyArray,energy
    
    def regressionChaos(self,data,a,b):
        return a+(b*data)
    
    def chaos01Test(self,data,c=np.pi/2,ncut=152,fs = 1526):
        
        if fs:
            ts = np.arange(0, len(data)/fs, 1/fs)
        else:
            ts = np.arange(0,len(data))
        ts = np.arange(0,len(data))
        N = len(data)
        p = np.zeros((N,))
        q = np.zeros((N,))
        #A little silly I realize, 
        p[0] = data[0]*np.cos(c)
        q[0] = data[0]*np.sin(c)
        M = np.zeros((ncut,))
        
        for ck in np.arange(1,N-1):
            p[ck] = p[ck-1] + (data[ck-1]*np.cos(c*ts[ck-1]))
            q[ck] = q[ck-1] + (data[ck-1]*np.sin(c*ts[ck-1]))
        #Get mean-squared displacement
        
        for jk in range(ncut):
            curSum = []
            for bc in range(N):
                if bc+jk >= len(data-1):
                    break
                else:
                    
                    curVal = np.power((p[bc+jk]-p[bc]),2) + np.power((q[bc+jk]-q[bc]),2) + 0.5*np.random.uniform(low=-1/2,high=1/2)
                    curSum.append(curVal)
            
            M[jk] = np.mean(curSum)
        #Get oscillation term
        
        expectedValData = np.mean(data)
        Vosc = np.zeros((ncut,))
        for tk in range(ncut):
            Vosc[tk] = np.power(expectedValData,2)*((1-np.cos(tk/fs*c))/(1-np.cos(c)))
        D = M - Vosc
        a = 1.1
        
        Dtilde = D-(a*np.min(D))
        Dtilde = Dtilde[1:len(Dtilde)]
        [Kc, pcov] = curve_fit(self.regressionChaos,np.log(np.arange(1,ncut)),np.log(Dtilde+0.0001))
        pdb.set_trace()
        return Kc[1]

    def estimateChaos(self,data,pltFlag = 0):
        numtrials = 100
        carray = np.random.uniform(low=np.pi/5,high = 3*np.pi/5,size = numtrials)
        #carray = np.arange(0.01,2*np.pi,0.01)
        Kest = np.zeros(len(carray),)
        for ck in range(len(carray)):
            Kest[ck] = self.chaos01Test(data,c=carray[ck])
        if pltFlag == 1:
            plt.plot(carray,Kest)
            plt.show()
        return np.median(Kest)
    
    def getZ(self,data):
        mArray = data[0:305]
        aMean = np.mean(mArray)
        aSDDV = np.std(mArray)
        ZScore = (data-aMean)/aSDDV
        return ZScore
    def getZBehind(self,data):
        mArray = data[1526-305:1526]
        aMean = np.mean(mArray)
        aSDDV = np.std(mArray)
        ZScore = (data-aMean)/aSDDV
        return ZScore
    
    def estimateEntropy_Continuous(R,RS,doQE=True):
        from entropy_estimators import continuous
        # compute the entropy using the k-nearest neighbour approach
        # developed by Kozachenko and Leonenko (1987):
        HR = continuous.get_h(R, k=5)
        HRS = continuous.get_h(RS,k=5)
        MIS = HR-HRS
        if doQE:
            #MI_Shuff = np.zeros((useTrials,))
            fractions = np.array([1, .5, .5, .5, .5, .25, .25, .25, .25, .25, .25, .25, .25])
            [nRows,nTrials] = np.shape(R)
            [nRows,nTrialsPerS] = np.shape(RS)
            useTrials = round(fractions*nTrials)
            useTrialsPerS = round(fractions*nTrialsPerS)
            partMIestimates = np.zeros((useTrials,))
            for ck in range(useTrials):
                #Rand perm around the rows
                sR = HR[np.arange(len(HR))[:,None], np.random.randn(*HR.shape).argsort(axis=1)]
                sRS = HRS[np.arange(len(HRS))[:,None], np.random.randn(*HRS.shape).argsort(axis=1)]
                RShuff = continuous.get_h(sR,k=5)
                RSShuff = continuous.get_h(sRS,k=5)
                partMIestimates[ck] = RShuff-RSShuff
            [p,S,mu] = np.polyfit(1./useTrialsPerS,partMIestimates,2)
            MIS = np.polyval(p,0)
        return MIS






    

                
                    


        







            

        

                








    



            

        
        


                    

                









        
        