#-------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry
# Date: 05/28/29                       Lovely, cloudy, London-like day in wisco
# Revision Hist: See github
# Purpose: Batch analysis of CSD data.
# Ref for Brandon: https://github.com/fmi-faim/napari-psf-analysis/blob/3c56c6aa8d22d586a045ac0698d759f2bfbba5dc/scratchpad/PSF%20Fitting.ipynb
#-------------------------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import matplotlib.cm as cm
#Let's do some defs:
def make_plot(xx, yy, zz, title='CSD', cmap=cm.bwr):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    t_max = np.max(np.abs(zz))
    levels = np.linspace(-1 * t_max, t_max, 32)
    im = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ticks = np.linspace(-1 * t_max, t_max, 3, endpoint=True)
    plt.colorbar(im, orientation='horizontal', format='%.2f', ticks=ticks)
    return ax
def parsePW(data):
        
        str1 = data.find("PW")
        str2 = data.find("PU")
        totStr = data[str2:str1+1]
        SplitSTR = totStr.split("_")
        if len(SplitSTR) == 2:
            val = SplitSTR[1]
            val = float(val[0:val.find("P")])
        elif len(SplitSTR) == 3:
            val = SplitSTR[2]
            if val == '1P':
                val = 0.1
            elif val == '2P':
                val = 0.2
            elif val == '3P':
                val = 0.3
            elif val == '4P':
                val = 0.4
            elif val == '5P':
                val = 0.5
            elif val == '6P':
                val = 0.6
            elif val == '7P':
                val = 0.7
            elif val == '8P':
                val = 0.8
            elif val == '9P':
                val = 0.9
            else:
                print('Error in conversion')
        else:
            print(data + ' Error')
        return val
df1 = pd.read_pickle('Z://PhDData//INSdata//LFPCSD0.pkl')
df2 = pd.read_pickle('Z://PhDData//INSdata//LFPCSD1.pkl')
df3 = pd.read_pickle('Z://PhDData//INSdata//LFPCSD3.pkl')

frames = [df1, df2, df3]
df = pd.concat(frames, ignore_index=True)
df = df.drop(df[df.DataID == 'INS2008'].index)       #Exclude because he recieved Michigan probe
df.reset_index(drop=True, inplace = True)
#df = pd.read_pickle('testCSD.pkl')
[numrows,numcols] = np.shape(df)
#numrows = 1
EPP = df['EnergyPerPulse']
uniqueEnergy = [0,0.5,1,1.5,2,2.5,3,3.5]
ISI = df['ISI']
xarray = df['xarray']
deltaX = xarray[0][1]-xarray[0][0]
deltaX = deltaX[0]
deltaY = 0.00375
xLocs = [0,14,29,43,57,71,86,100]           #np.where(np.abs(xarraylin-1.75)==np.min(np.abs(xarraylin-1.75))), xarraylin=np.arange(0,1.75+0.0175,0.0175)
xHalf = int(np.floor(0.125/deltaX))         #1 sided half distance between X electrode locations
yLocs = [0,100]
yHalf = int(np.floor(0.1875/0.00375))
yarray = df['yarray']
maxSink = {}
maxSource = {}
dfSink = pd.DataFrame(columns=['EnergyPerPulse','maxSink'])
dfSource = pd.DataFrame(columns=['EnergyPerPulse','maxSource'])
elecX = np.array((0,0.25,0.5,0.75,1,1.25,1.5,1.75))
elecY = np.array((0,0.375))
#sources = np.zeros((101,101,numrows))
#sinks = np.zeros((101,101,numrows))
dfCSD = pd.DataFrame(columns=['DataID', 'EnergyPerPulse','ISI','Sources','Sinks'])
for jk in range(len(uniqueEnergy)):
    maxSink[str(uniqueEnergy[jk])] = []
    maxSource[str(uniqueEnergy[jk])] = []
for ck in range(numrows):
    PW = parsePW(df.DataID[ck])
    ISI = df.ISI[ck]
    nP = df.NPulses[ck]
    stimTime = (nP*PW)+((nP-1)*ISI)
    #Stim window + 50 ms
    stimSamples = int(np.round(stimTime*(1526/1000))+77)
    curCSD = df['estCSD'][ck]
    maxCurrentSinkVec = np.empty(stimSamples,)
    maxCurrentSourceVec = np.empty(stimSamples,)
    stimRangeVec = np.arange(305,305+stimSamples)
    for bc in range(len(stimRangeVec)):
        curcurCSD = curCSD[:,:,stimRangeVec[bc]]
        maxCurrentSinkVec[bc] = np.min(curcurCSD)
        maxCurrentSourceVec[bc] = np.max(curcurCSD)
    maxCurrentSink = np.min(maxCurrentSinkVec)
    maxCurrentSource = np.max(maxCurrentSourceVec)
    sinkWhere = np.where(maxCurrentSinkVec==maxCurrentSink)
    sourceWhere = np.where(maxCurrentSourceVec==maxCurrentSource)
    sinkWhere = sinkWhere[0][0]
    sourceWhere = sourceWhere[0][0]
    modCSD = curCSD[305:305+stimSamples]
    sources = modCSD[:,:,sourceWhere]
    sinks = modCSD[:,:,sinkWhere]
    if EPP[ck] <0.5:
        curEPP = str(uniqueEnergy[0])
    elif 0.5<= EPP[ck] < 1: 
        curEPP = str(uniqueEnergy[1])
    elif 1<= EPP[ck] < 1.5: 
        curEPP = str(uniqueEnergy[2])
    elif 1.5<= EPP[ck] < 2: 
        curEPP = str(uniqueEnergy[3])
    elif 2<= EPP[ck] < 2.5: 
        curEPP = str(uniqueEnergy[4])
    elif 2.5<= EPP[ck] < 3: 
        curEPP = str(uniqueEnergy[5])
    elif 3<= EPP[ck] < 3.5: 
        curEPP = str(uniqueEnergy[6])
    elif EPP[ck]>=3.5: 
        curEPP = str(uniqueEnergy[7])
    maxSink[curEPP].append(maxCurrentSink)
    maxSource[curEPP].append(maxCurrentSource)
    dfSink.loc[-1] = [curEPP,maxCurrentSink]
    dfSource.loc[-1] = [curEPP,maxCurrentSource]
    dfSink.index = dfSink.index + 1  # shifting index
    dfSink = dfSink.sort_index()  # sorting by index
    dfSource.index = dfSource.index + 1  # shifting index
    dfSource = dfSource.sort_index()  # sorting by index
    [xSink,ySink,tSink] = np.where(curCSD==maxCurrentSink)
    [xSource,ySource,tSource] = np.where(curCSD==maxCurrentSource)
    xSink = xSink[0]
    ySink = ySink[0]
    tSink = tSink[0]
    xSource = xSource[0]
    ySource = ySource[0]
    tSource = tSource[0]
    dfCSD.loc[-1] = [df.DataID[ck],EPP[ck],ISI,sources,sinks]
    dfCSD.index = dfCSD.index + 1  # shifting index
    dfCSD = dfCSD.sort_index()  # sorting by index

labelsSink, dataSink = [*zip(*maxSink.items())]
labelsSource, dataSource = [*zip(*maxSource.items())]
plt.figure()
plt.boxplot(dataSink,vert=False)
#plt.xticks(range(1, len(labelsSink) + 1), labelsSink)
plt.boxplot(dataSource,vert=False)
#plt.xticks(range(1, len(labelsSource) + 1), labelsSource)
plt.show()
pdb.set_trace()
dfCSD.to_pickle('CSDTrain.pkl')