# -*- coding: utf-8 -*-
# %load ./../functions/detect_cusum.py
"""
Created on Wed Jun 27 16:21:11 2018

@author: lx2347

Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

"""

import numpy as np
import matplotlib.dates as mdates

def detect_cusum(time, x, save_dir, threshold=1, drift=0,  
                 ending=False, show=True, save = False, ax=None):
    
    """
    Parameters
    ----------
    time : 1D array-like 
        time. 
    x : 1D array_like
        data.
    save_dir: string 
        location to save plots 
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).

    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.

    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.

    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.

    See this IPython Notebook [2]_.

    References
    ----------
    .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
    .. [2] hhttp://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

    """
 
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    gp_real, gn_real = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
     
#    # Find changes (online form)
#    for i in range(1, x.size):
#        s = x[i] - x[i-1]
#        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
#        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
#        if gp[i] < 0:
#            gp[i], tap = 0, i
#        if gn[i] < 0:
#            gn[i], tan = 0, i
#        if gp[i] > threshold or gn[i] > threshold:  # change detected!
#            ta = np.append(ta, i)    # alarm index
#            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
#            gp[i], gn[i] = 0, 0      # reset alarm
#    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Find changes (online form)
    # add gp_real and gn_rel to track the actual amplitude of the transient
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gp_real[i] = gp_real[i-1] + s 
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        gn_real[i] = gn_real[i-1] - s 
        
        if gp[i] < 0:
            gp[i], gp_real[i], tap = 0, 0, i
        if gn[i] < 0:
            gn[i], gn_real[i], tan = 0, 0, i
        if gp_real[i] > threshold or gn_real[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp_real[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
            gp_real[i], gn_real[i] = 0, 0
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE
    
    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _,_,_ = detect_cusum(time[::-1],x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]
           
    if show:
        _plot(time, x, threshold, drift, ending, ax, ta, tai, taf, 
              gp_real, gn_real,save,save_dir)

    return ta, tai, taf, amp, gp_real, gn_real 


def _plot(time, x, threshold, drift, ending, ax, ta, tai, taf, gp, gn, save, save_dir ):
    """Plot results of the detect_cusum function, see its help."""

    try:
        
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            
        hfmt = mdates.DateFormatter('%H:%M:%S')
        ax1.plot_date(time , x, 'k-', lw=2,tz="US/Central")
        ax1.plot_date(raw_time, raw_signal, 'b-', alpha =0.3, lw=2,tz="US/Central")
        if len(ta):
            ax1.plot_date(time[tai], x[tai], '>', mfc='g', mec='g', ms=10,
                     tz="US/Central",label='Start')

            if ending:
                ax1.plot_date(time[taf], x[taf], '<', mfc='g', mec='g', ms=10,
                         label='End',tz="US/Central")
#            ax1.plot_date(time[ta], x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
#                     label='Alarm',tz="US/Central")
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim([time[0], time[-1]])
        ax1.set_xlabel('Time [h:min:s]', fontsize=14)
        ax1.set_ylabel('Pressure [psi]', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('(a)', fontsize=14)
#        ax1.set_xticks([time[0], time[-1]])
#        ax1.xaxis.set_major_locator(mdates.HourLocator())
        ax1.xaxis.set_major_formatter(hfmt)
#        ax1.set_title('Time series and detected pressure transients ' +
#                      '(threshold= %.3g, drift= %.3g): N changes = %d'
#                      % (threshold, drift, len(tai)))
        ax2.plot_date(time, gp, 'y-', label='+',tz="US/Central")
        ax2.plot_date(time, gn, 'm-', label='-',tz="US/Central")
        ax2.set_xlim([time[0], time[-1]])
        ax2.set_xlabel('Time', fontsize=14)
        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color='r')
        ax2.set_ylabel('Cumulative Sum (c)', fontsize=14)
#        ax2.set_title('Time series of the cumulative sums of ' +
#                      'positive and negative changes')
        ax2.set_title('(b)', fontsize=14)
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        ax2.xaxis.set_major_formatter(hfmt)
        plt.tight_layout()
        plt.show()
        
        #save 
        if save :
            fig.savefig(save_dir + 'TransientDetected.png', format='png', dpi=100)
            
            
def CountNumberofContinuousChange( tai,taf):

    if len(tai) ==len(taf):
        n = len(tai)
    else:
        print("Length of start time nt equal to the length of end time")
        
    ContiNum = []
    conti = 1  
    tai = np.append(tai,float('inf'))
    for i in range(0,n):
        if tai[i+1] - taf[i] <= 1:

            conti = conti + 1 
        else:
            ContiNum.append(conti)               
            conti = 1

    return ContiNum
    
    
            
    