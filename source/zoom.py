# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:00:28 2018

@author: lx2347
"""

# number processing 
import numpy as np
# plot 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



def zoom (time, signal, raw_time, raw_signal, freq, 
          tai, taf, ta, amp, interval, resolution, zooms, save_dir, 
           show = True , save = True):
    
    n = len(tai)
    
    nzoom = 1 
    zoom_id = np.array([tai[0]])
    zoom_end_id = np.array([taf[0]])
    i_new = 0 
    for i in np.arange(0, n-1):
        # if the end point of the next transient falls out of the "4min" window, 
        # open another window for the next transient
        if taf[i+1] >tai[i_new] + interval :
            nzoom += 1
            zoom_id = np.append(zoom_id,tai[i+1])
            zoom_end_id = np.append(zoom_end_id,taf[i+1])
            i_new = i+1
            
    dp = np.zeros(nzoom)  
    zoom_amplitude= np.zeros(nzoom)         
#    zoom_start_time = np.zeros(nzoom) 
    zoom_start_time = [0] * nzoom
#    #plot 
#    if show :          
#        fig = plt.figure(figsize=(12, nzoom*3), dpi=80, facecolor='w', edgecolor='k')        

    for j in np.arange(nzoom):
        
        """
        If the transient if longer than 4min, plot the whole transient
        If not, open a 4min window and plot time series within the window,
        which may includes some other transient. 
        """
        begin = int(zoom_id[j])
#            if zoom_end_id[j] < zoom_id[j] + interval :
#                end = int(zoom_id[j] + 1 +interval)
#            else:
#                end = int(zoom_end_id[j] +1 )
        end = int(zoom_id[j] + 1 +interval)
            
        # search the start and end point within the plotting window    
        zoom_index = np.where(np.logical_and(tai>=begin, taf<end))
        zoom_tai = tai[zoom_index]
        zoom_taf = taf[zoom_index]
        zoom_ta = ta[zoom_index]
        
        # project back to the index of raw data 
        begin_raw = int((zoom_id[j] +0.5)*freq)
#            if zoom_end_id[j] < zoom_id[j] + interval :
#                end_raw = int((zoom_id[j] +0.5+ interval)*freq)
#            else:
#                end_raw = int((zoom_end_id[j] +0.5 )*freq)
        end_raw = int((zoom_id[j] +0.5+ interval)*freq)
        zoom_time = time[begin : end]
        

        zoom_signal = signal[begin : end]
        
        zoom_raw_time = raw_time[begin_raw:end_raw]
        zoom_raw_signal = raw_signal[begin_raw:end_raw]
        
        amp = max(zoom_signal)- min(zoom_signal)
        amp_raw = max(zoom_raw_signal) - min(zoom_raw_signal)
        
        dp[j] = np.absolute(zoom_signal[-2]- zoom_signal[-1])           
        zoom_amplitude[j] = max(zoom_signal) - min(zoom_signal)
        
        zoom_start_time[j] =  mdates.num2date(time[begin])
#       zoom_start_time[j] =  mdates.num2date(time[begin]).hour
        # add all the zoomed asignal with the same length to the zoom database
        if len(zoom_signal) == interval+1:
            zooms.append(zoom_signal)
        #plot 
        if show :  
            fig = plt.figure(figsize=(12, nzoom*3), dpi=80, facecolor='w', edgecolor='k')  
            hfmt = mdates.DateFormatter(' %H:%M:%S')
            ax = fig.add_subplot(nzoom, 1, j+1)           
            ax.plot_date(zoom_time, zoom_signal,  mfc='k', mec='k',ms=3, tz="US/Central",
                     xdate=True, label= '%d s' %resolution)
            ax.plot_date(zoom_raw_time, zoom_raw_signal, 'b-', tz="US/Central",
                     xdate=True, label ='64hz')
#        tz=pytz.timezone("US/Central")    
            ax.plot_date(time[zoom_tai], signal[zoom_tai], '>', mfc='g', mec='g', ms=10,
                     tz="US/Central",label='Start')
            ax.plot_date(time[zoom_taf], signal[zoom_taf], '<', mfc='g', mec='g', ms=10,
                     tz="US/Central",label='End')
#            ax.plot_date(time[zoom_ta], signal[zoom_ta], 'o', mfc='r', mec='r', mew=1, ms=5,
#                     label='Alarm',tz="US/Central")
            ax.set_xlim([zoom_time[0], zoom_time[-1]])
            ax.xaxis.set_major_formatter(hfmt)
            ax.legend(loc='best', framealpha=.5, numpoints=1)
            plt.xlabel('Time')
            plt.ylabel('Pressure (psi)') 
    
#            if signal[begin] < signal[begin+1]:
#                plt.text(0.7,0.8, 'Uploading', transform=ax.transAxes, fontsize=12)
#                plt.text(0.45 , 0.6, 
#                         'Transient amplitude = %.3g psi, Transient raw amplitude = %.3g psi' 
#                         % (amp, amp_raw), transform=ax.transAxes, fontsize=10)
#            else:
#                plt.text(0.7,0.3, 'Under-loading', transform=ax.transAxes, fontsize=12)
#                plt.text(0.45 , 0.1, 
#                         'Transient amplitude = %.3g psi, Transient raw amplitude = %.3g psi' 
#                         % (amp, amp_raw), transform=ax.transAxes, fontsize=10)
    
    plt.suptitle("Detailed Plot of Pressure Transient")
    plt.show()
#    zooms = np.array(zooms)    
    #save 
    if save :
        fig.savefig(save_dir + 'DetailedTransient.png', format='png', dpi=600)
    return nzoom, zooms, dp,zoom_amplitude,zoom_start_time
        