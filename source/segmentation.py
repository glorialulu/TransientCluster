# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:07:06 2018
@author: lx2347

This code intends to apply the following segementation scheme to the given time 
series data. 

1. Piecewise Aggregate Approximate (PAA)
2. 
"""
import matplotlib.pyplot as plt
import numpy as np


def PiecewiseAggregateApproximate(time,signal,every,save_dir, show = False, save= False):
       
    original_size = len(signal)

    paa_size = int(original_size / every)

    if (every == 1 ):
        paa_signal = signal 
    else:
        paa_signal = np.zeros(paa_size)
        paa_time = np.zeros(paa_size)
        i = 1
        while i <= paa_size :
            paa_signal[i-1] = np.average(signal[every*(i-1) : every*i-1])
            paa_time[i-1] = np.average(time[every*(i-1): every*i-1])
            i =i+1
        # plot 
        if show :
            fig =plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k') 
            plt.plot_date(time, signal, 'b-', alpha=0.5 ,tz="US/Central")
            plt.plot_date(paa_time, paa_signal,'r-',tz="US/Central")
            plt.title("Dimension Reduction "+
                      'sampling interval= %.3g s' % (every/64))   
        
        # save 
        if save :
            fig.savefig(save_dir + 'PAA_ReduceDim'+str(int(every/64))+ '.png', format='png', dpi=600)
        
    return paa_time, paa_signal

PAA = PiecewiseAggregateApproximate 

#
#
## The following codes are for testing  
#    
#def read_data_frame(filename):
#    headers = ['CT','UTC','Epoch','Pressure']
#    df = pd.read_csv(filename,names=headers)
#    df= pd.DataFrame(df)
#    return df
#
#
#df=read_data_frame('STN26_20180520-20180521.csv')
#x=df['UTC'].tolist()
#x.pop(0)
#for i in range(len(x)):
#    x[i]=datetime.strptime(str(x[i]), '%Y/%m/%d %H:%M:%S.%f')
#y=df['Pressure'].tolist()
#y.pop(0)
#x = matplotlib.dates.date2num(x)
###df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%d/%m/%y %H:%M:%S.%f'))
###x = df['Date'].tolist()
###y = df['Pressure'].tolist()
#plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
#plt.plot_date(x,y,'b-')
#plt.xlabel('Time')
#plt.ylabel('Pressure (psi)')
##plt.xticks(rotation=20)
#plt.title("Pressure Plot ")
#plt.show()
#
#signal = np.array(y,dtype=np.float32)
#time = np.array(x)
#sampling_frequency = 64
#samle_every_second = 30  # try to keep this as multiple of 64 
#every = sampling_frequency * samle_every_second
#PiecewiseAggregateApproximate(x, signal, every)