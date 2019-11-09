# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:09:13 2018

@author: lx2347
"""
import os
import numpy as np
# plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
# date
import time
# distance matrix to n-D coordinates
from sklearn.manifold import MDS
#Lu's module
from preprocessing import preprocessing
import segmentation as seg
from detect_cusum import detect_cusum
from detect_cusum import CountNumberofContinuousChange
from zoom import zoom
import clustering

#########################################################################
#---- [USER INPUT] ----
start_time = time.time()
raw_freq = 64  # frequency of the raw data
save_dir = '..\\results' # the directory to save results
# location of the hdf5 files or csv files
hdf5_root = '..\\sample_data\\hdf5\\'
stn = 'austin_21'  # name of the station

nplots = []
nchanges = []
ntimes = []

ContinousChange = []

tai_list = []
taf_list = []
amp_list = []
tai_hour_list = []
change_list = []

zooms = []
dp_list = []
zoom_start_list =[]
zoom_amp_list = []
transient_list = []

for filename in sorted(os.listdir(os.path.join(hdf5_root, stn))):
    print('='*50)
    print("Working on file" , filename ,"......")
    file = os.path.join(hdf5_root, stn, filename)

    # preprocessing the data
    print ("Preprocessing the data ... ")
    x, signal = preprocessing(file, save_dir, plot = False)


    nplot = []
    nchange = []
    ntime = []
    # Reduce the dimension
    resolutions = np.array([10])

    for resolution in resolutions : # in units of second
#%%
# =============================================================================
# Segregation
# =============================================================================
        start_time = time.time()
        print("Segmenting the original data...")

        sampling_freq = raw_freq * resolution
        reduced_time, reduced_signal = seg.PAA(x, signal, sampling_freq, save_dir,
                                           show=False, save = False)

        smooth_signal = reduced_signal

        # calculate the mean and standard deviation
        mean = np.average(smooth_signal)
        std = np.std(smooth_signal)

        # parameters for cusum algorithm
        threshold = 10
        drifts = np.array([threshold/100 *resolutions])
        for drift in drifts :
#%%
# =============================================================================
# Detecting
# =============================================================================
            print ("Detecting Transients..." )
            ta, tai, taf, amp,_,_ = detect_cusum( reduced_time, smooth_signal, save_dir,
                                 threshold, drift, ending=True,
                                 show= False, save = False, ax=None)

            nchange = np.append(nchange, len(ta))
            tai_list+= reduced_time[tai].tolist()
            taf_list+= reduced_time[taf].tolist()
            amp_list+= amp.tolist()
            tai_hour_list += [i.hour for i in mdates.num2date(reduced_time[tai])]

            conti = CountNumberofContinuousChange(tai,taf)
            ContinousChange.append(conti)


#%%
# =============================================================================
# Zooms in transient
# =============================================================================
            # If any change is detected, let's then zoom in the transient areas
            print("Zooming in the detected transient with window-second window...")
            nzoom = 0
            window = 300   #in unit of second
            if len(ta) > 0 :
                interval = window/resolution  # set window size
                nzoom, zooms, dp, zoom_amplitude, zoom_start_time = zoom(reduced_time,
                                smooth_signal, x, signal, sampling_freq,
                                tai, taf, ta, amp, interval, resolution, zooms, save_dir,
                                show= False , save = False  )

                nplot = np.append(nplot, nzoom)

                dp_list+= dp.tolist()
                zoom_amp_list += zoom_amplitude.tolist()
                zoom_start_list += zoom_start_time

            # record the computation time
            end_time = time.time()
            ntime = np.append(ntime, end_time - start_time)

#%%
# =============================================================================
# Write and Save
# =============================================================================
# write information about changes into change_list
duration = np.array(taf_list) - np.array(tai_list)
duration = duration* 86400
duration_list = duration.tolist()
change_list = {"Start_time": tai_list,
                  "End_time"  : taf_list,
                  "Amplitude" : amp_list,
                  "Duration"  : duration_list,
                  "Pressure_Difference": dp_list,
                  "tai_hour": tai_hour_list}

print('Total Number of change detected= ', len(tai_list))

## write change_list to csv
with open(save_dir + '\\change_list_' + stn + '.csv', "w", newline='') as outfile:
   writer = csv.writer(outfile)
   writer.writerow(change_list.keys())
   writer.writerows(zip(*change_list.values()))

#write information about transients into transient_list
transient_list = {"Start_time": zoom_start_list,
                  "Amplitude" : zoom_amp_list}

print('Total Number of transients formed =', len(zooms))

## write transient_list to csv
with open(save_dir + '\\transient_list_' + stn + '.csv', "w", newline='') as outfile:
   writer = csv.writer(outfile)
   writer.writerow(transient_list.keys())
   writer.writerows(zip(*transient_list.values()))

#%%
# =============================================================================
# Normalize the detected signals
# =============================================================================
zooms_stnd = np.copy(zooms)
#Normalize individually
for i in range(len(zooms)):
    zooms_stnd[i] = (zooms[i]- np.min(zooms[i]))/ (np.max(zooms[i]) - np.min(zooms[i]))

#%%
## =============================================================================
## DTW distance before clustering
## =============================================================================
w = 5  # window size of DTW distance
dist_mat = np.zeros((len(zooms),len(zooms)))

f = open(save_dir +'\\dtw_matrix.txt','w')
for signal1_ind in range(len(zooms)-1):
    for signal2_ind in range(signal1_ind+1,len(zooms)) :

        signal1 = zooms_stnd[signal1_ind]
        signal2 = zooms_stnd[signal2_ind]
        dist_mat[(signal1_ind,signal2_ind)] = clustering.DTWDistance(signal1, signal2,w)
        dist_mat[(signal2_ind,signal1_ind)] = clustering.DTWDistance(signal1, signal2,w)
        f.write(str(signal1_ind+1)+" ")
        f.write(str(signal2_ind)+" ")
        f.write(str(dist_mat[(signal1_ind,signal2_ind)])+"\n" )
f.close()

# %%
## =============================================================================
## Multidimensional scaling
## =============================================================================
embedding = MDS(n_components=2, eps=1e-6, max_iter=2000
                ,dissimilarity="precomputed", random_state=6)
#embedding = TSNE(n_components=2, metric="precomputed", random_state=6)
projct2d= embedding.fit_transform(dist_mat)
# plot 2d space
fig= plt.figure(figsize=(4,4), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(projct2d[:,0], projct2d[:,1],marker='o', c="k", alpha=1, edgecolor='k')

plt.xlabel("Distance space #1")
plt.ylabel("Distance space #2")
plt.show()
#save
fig.savefig(save_dir + '\\DistanceSpace.png', format='png', dpi=600)


#%%
## =============================================================================
## K-means clustering
## =============================================================================

print('Clustering ...')
randm_init = 10
max_iter = 100

num_cluster_list = range(2,3) # the list of number of clusters
SSE_list = np.zeros(len(num_cluster_list))
sil_list = np.zeros(len(num_cluster_list))
ch_list = np.zeros(len(num_cluster_list))

i=0
for num_cluster in num_cluster_list:
#  k-means
    centroid, assignment, cost,sorted_zooms, sil_avg_kmeans, SSE_sum_kmeans, ch_score_kmeans = clustering.k_means (
         zooms_stnd, projct2d, dist_mat, num_cluster, max_iter, randm_init, interval +1,
          w,  plot = True, sil = True, SSE = True, sort = True )
    print('K-means clustering evaluation: Silhouettes Index = %.3g; SSE= %.3g, C-H sore =%.3g'
               %(sil_avg_kmeans, SSE_sum_kmeans,  ch_score_kmeans))
    SSE_list[i] = SSE_sum_kmeans
    sil_list[i] = sil_avg_kmeans
    ch_list[i] = ch_score_kmeans
    i +=1


#%%
# =============================================================================
# Plot scores
# =============================================================================

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, host = plt.subplots(figsize=(7,4), dpi=500, facecolor='w', edgecolor='k')
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

p1, = host.plot(num_cluster_list, SSE_list, '-go', label = 'SSE')
p2, = par1.plot(num_cluster_list, sil_list, '-r*', label = 'Silhouette coefficient')
p3, = par2.plot(num_cluster_list, ch_list, '-bs', label = 'Calinski-Harabaz index')

host.set_xlim(num_cluster_list[0], num_cluster_list[-1])

host.set_xlabel('Number of clusters ')
host.set_ylabel('Sum of Squared Error')
par1.set_ylabel('Average silhouette Score')
par2.set_ylabel("Calinski-Harabaz Index")

tkw = dict(size=4, width=1.5)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines], loc='best')
host.grid()
plt.show()
fig.savefig(save_dir + '\\valuation_3c.pdf', format='pdf', dpi=1000)
