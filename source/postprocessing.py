# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:23:52 2018

@author: lx2347
"""
import numpy as np 
# plot 
import matplotlib.pyplot as plt

def postprocessing(resolutions, nplots, nchanges, ntimes,stn ):
    
    n = np.shape(nplots)[0]
    print(n)
    nplots_ave = np.average(nplots, axis = 0 )
    nplots_std = np.std(nplots, axis = 0 )
    nplots_up = nplots_ave + 1.96 *nplots_std/ np.sqrt(n)
    nplots_low = nplots_ave - 1.96 *nplots_std/ np.sqrt(n)
    
    nchanges_ave = np.average(nchanges, axis = 0 )
    nchanges_std = np.std(nchanges, axis = 0 )
    nchanges_up = nchanges_ave + 1.96 *nchanges_std/ np.sqrt(n)
    nchanges_low = nchanges_ave - 1.96 *nchanges_std/ np.sqrt(n)

    ntimes_ave = np.average(ntimes, axis = 0 )
    ntimes_std = np.std(ntimes, axis = 0 )
    ntimes_up = ntimes_ave + 1.96 *ntimes_std/ np.sqrt(n)
    ntimes_low = ntimes_ave - 1.96 *ntimes_std/ np.sqrt(n)

    #Plot PAA resolution study
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 9))
    ax1.plot(resolutions, nplots_ave,'r-+')
    ax1.plot(resolutions, nplots_up,'b--')
    ax1.plot(resolutions, nplots_low,'b--')
    ax1.set_xlabel('Resolution of PAA [s]')
    ax1.set_ylabel('Number of Transient Plot per day')
    
    ax2.plot(resolutions, nchanges_ave,'r-+')
    ax2.plot(resolutions, nchanges_up,'b--')
    ax2.plot(resolutions, nchanges_low,'b--')
    ax2.set_xlabel('Resolution of PAA [s]')
    ax2.set_ylabel('Number of Detected changes per day')
    
    ax3.plot(resolutions,ntimes_ave,'r-+')
    ax3.plot(resolutions, ntimes_up,'b--')
    ax3.plot(resolutions, ntimes_low,'b--')
    ax3.set_xlabel('Resolution of PAA [s]')
    ax3.set_ylabel('Computation Time')
    plt.tight_layout()
#    plt.suptitle("PAA Parameter Tunning at " + stn)
    fig.savefig( "PAA_Parameter_Tunning_at" + stn+ ".png", format='png', dpi=600)
