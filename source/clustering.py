# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:36:29 2018

@author: lx2347
"""
import math
import random
import numpy as np
# import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
# import hdbscan

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage

#plot
from itertools import cycle

import json
 #%%

def DTWDistance(s1, s2,w):
    """
    Computes Dynamic Time Warping (DTW) Distance between two sequences.
    ----------
     Parameters
    ----------
    s1 : 1-D array like
         signal 1
    s2 : 1-D array like
         signal 2
    w : int
        how many shifts are computed
    ----------
     Returns
    ----------
     the minimum distance
    """
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))
    w_elastic = np.zeros(len(s1))
    for i in range(len(s1)):
        w_elastic[i] = i/len(s1)*w

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        we = int(round(w_elastic[i]))
        for j in range(max(0, i-int(w)), min(len(s2), i+int(w)+1)):
            dist= (s1[i]-s2[j])**2

            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
 #%%

def DTWDistance_vis(s1, s2,w):
    """
    Computes Dynamic Time Warping (DTW) Distance between two sequences.
    ----------
     Parameters
    ----------
    s1 : 1-D array like
         signal 1
    s2 : 1-D array like
         signal 2
    w : int
        how many shifts are computed
    ----------
     Returns
    ----------
    dist : real
           the minimum distance
    acc : len(s1) * len(s2) array
          the accumulated cost matrix
    path : 2-D array
           the wrap path

    """

    DTW={}


    w = max(w, abs(len(s1)-len(s2)))
#    w_elastic = np.zeros(len(s1))
#    for i in range(len(s1)):
#        w_elastic[i] = i/len(s1)*w

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    dis = np.zeros((len(s1),len(s2)))
    for i in range(len(s1)):
#        we = int(round(w_elastic[i]))
        for j in range(max(0, int(round(i-w))), min(len(s2), int(round(i+w+1)))):
            d= (s1[i]-s2[j])**2
            dis[i,j]= d
            DTW[(i, j)] = d + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    dist = np.sqrt(DTW[(len(s1)-1, len(s2)-1)])

    acc = np.array(list(DTW.values()), dtype= 'float32' )
    tmp = acc.reshape(len(s1)+1,len(s2)+1)
    acc = tmp[1:, 1:].reshape(len(s1),len(s2))

    if len(s1)==1:
         path = np.zeros(len(s2)), range(len(s2))
    elif len(s2) == 1:
         path = range(len(s1)), np.zeros(len(s1))
    else:
         path = _traceback(tmp)

# another yet more explicit way of calculating path
#    path = [[len(s1)-1, len(s2)-1]]
#    i = len(s1)-1
#    j = len(s2)-1
#    while i>0 and j>0:
#        if i==0:
#            j = j - 1
#        elif j==0:
#            i = i - 1
#        else:
#            if acc[i-1, j] == min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1]):
#                i = i - 1
#            elif acc[i, j-1] == min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1]):
#                j = j-1
#            else:
#                i = i - 1
#                j= j- 1
#        path.append([j, i])
#    path.append([0,0])
#    print(path)
    return dist, acc, path, dis

#%%
def visualize_DTW(zooms, w):
    dist_mat = np.zeros((len(zooms), len(zooms)))
    for signal1_ind in range(len(zooms)):
        for signal2_ind in range(len(zooms)) :
            # Standardization
            signal1 = zooms[signal1_ind]
            signal2 = zooms[signal2_ind]

            dist, acc, path, d  = DTWDistance_vis(signal1, signal2,w)

            dist_mat[signal1_ind,signal2_ind] = dist
            x = np.arange(len(signal1))
            y = np.arange(len(signal2))

            X, Y = np.meshgrid(x, y)

            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                    cmap(np.linspace(minval, maxval, n)))
                return new_cmap

            cmap = cm.Oranges_r
            new_cmap = truncate_colormap(cmap, 0, 0.7)
#            plot eulidian distance
            fig, axScatter = plt.subplots(figsize=(8, 8))
            axScatter.imshow(d.T, origin='lower',
                                    cmap=new_cmap, interpolation='nearest')
#            # add values
#            for i in range(len(signal1)):
#                for j in range(len(signal2)):
#                    plt.text(i, j, round(d[i, j],2),
#                       ha="center", va="center", color="w")

            axScatter.grid(True)
#            axScatter.set_title('Dynamic Time Wrapping Distance = %.3g' %dist)
#            axScatter.set_ylabel('Signal %i' % signal2_ind )
            axScatter.set_ylabel('Signal X2',fontsize = 14  )
            axScatter.yaxis.set_label_position("right")
#            axScatter.set_xlabel('Signal %i' % signal1_ind)
            axScatter.set_xlabel('Signal X1' ,fontsize = 14)
            divider = make_axes_locatable(axScatter)
            ax1 = divider.append_axes("top", 1.2, pad=0.5)
            ax2 = divider.append_axes("left", 1.2, pad=0.5)
            ax1.plot(signal1,'k-*')
            ax1.set_xticklabels([])
            ax1.set_xlim(0, len(signal1)-1)
#            ax1.set_ylabel('Normalized Pressure (psi)')
#            ax1.set_title(('Signal %i' % signal1_ind ),fontsize = 12)
            ax1.set_title(('Signal X1' ),fontsize = 14)
            ax1.grid(True)

            ax2.plot(signal2,y,'k-*')
            ax2.set_yticklabels([])
            ax2.invert_xaxis()
            ax2.set_ylim(0, len(signal1)-1)
#            ax2.set_xlabel('Normalized Pressure (psi)')
#            ax2.set_ylabel(('Signal %i' % signal2_ind ),fontsize = 12)
            ax2.set_title(('Signal X2' ),fontsize = 14)
            ax2.grid(True)
#            fig.colorbar(show, ax=[ax1,ax2])
            plt.show()
            fig.savefig('EulidianMatrix.pdf', format='pdf', dpi=600)


#plot cumulative distance and wrapping path
            fig, axScatter = plt.subplots(figsize=(8, 8))
            axScatter.imshow(acc.T, origin='lower',
                                    cmap=new_cmap, interpolation='nearest')
#            plt.colorbar(show)
#            path_x = [point[0] for point in path]
#            path_y = [point[1] for point in path]

            axScatter.plot(path[0], path[1], 'w')
            axScatter.grid(True)
#            axScatter.set_title('Dynamic Time Wrapping Distance = %.3g' %dist)
#            axScatter.set_ylabel('Signal %i' % signal2_ind )
            axScatter.set_ylabel('Signal X2',fontsize = 14  )
            axScatter.yaxis.set_label_position("right")
#            axScatter.set_xlabel('Signal %i' % signal1_ind)
            axScatter.set_xlabel('Signal X1' ,fontsize = 14)
            # add values
#            for i in range(len(signal1)):
#                for j in range(len(signal2)):
#                    axScatter.text(i,j, round(acc[i, j],2),
#                       ha="center", va="center", color="w")
#            axScatter.text(len(signal2)-1, len(signal1)-1, round(dist**2, 2),
#                       ha="center", va="center", color="r",fontweight="bold")

            divider = make_axes_locatable(axScatter)
            ax1 = divider.append_axes("top", 1.2, pad=0.5)
            ax2 = divider.append_axes("left", 1.2, pad=0.5)
            ax1.plot(signal1,'k-*')
            ax1.set_xticklabels([])
            ax1.set_xlim(0, len(signal1)-1)
#            ax1.set_ylabel('Normalized Pressure (psi)')
#            ax1.set_title(('Signal %i' % signal1_ind ),fontsize = 12)
            ax1.set_title(('Signal X1' ),fontsize = 14)
            ax1.grid(True)

            ax2.plot(signal2,y,'k-*')
            ax2.set_yticklabels([])
            ax2.invert_xaxis()
            ax2.set_ylim(0, len(signal1)-1)
#            ax2.set_xlabel('Normalized Pressure (psi)')
#            ax2.set_ylabel(('Signal %i' % signal2_ind ),fontsize = 12)
            ax2.set_title(('Signal X2' ),fontsize = 14)
            ax2.grid(True)
#            fig.colorbar(show, ax=[ax1,ax2])
            plt.show()
            fig.savefig('path.pdf', format='pdf', dpi=600)
    return dist_mat
#%%

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


#%%
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)
#%%
def cal_cost (data, assignments, centroids, w ) :
    cost = 0
    for key in assignments:
        for k in assignments[key]:
            cost += DTWDistance(centroids[key], data[k], w)

    return cost
#%%
def sort_cluster(data, assignment):

    sorted_data = []
    for key in assignment:
        sorted_data += data[assignment[key]].tolist()
    return sorted_data



#%%
def plot_cluster_k(data, centroid, assignment, num_clust, method ) :

    fig = plt.figure(figsize=(6, num_clust*2), dpi=80, facecolor='w', edgecolor='k')
    pattern = 1
    for key in assignment:

        ax = fig.add_subplot(num_clust, 1, pattern)

        for k in assignment[key]:
            ax.plot(data[k], alpha=0.2 )
        pattern += 1
        ax.plot(centroid[key], 'k-+', linewidth=3.0, label ='pattern %d' %key )
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure [psi]')
        ax.set_xlim([0, len(centroid[key])-1 ])
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    plt.suptitle("Transient Patterns clustered by %s Algorithm" %method)
    plt.show()
    fig.savefig('..\\results\\Transient_patterns_clustered.pdf', format='pdf', dpi=1000)
#%%
def transient_intensity(data, assignment) :


    intensity ={}

#    for key in assignment:
#
#        for k in assignment[key]:
#            if key in intensity:
#                intensity[key].append(max(data[k]-min(data[k])))
#            else:
#                intensity[key] = []
#                intensity[key].append(max(data[k]-min(data[k])))
##        bins = np.linspace(min(intensity[key]), max(intensity[key]), 20)
#        bins = np.linspace(0, 75, 30)
#        hist,_= np.histogram(intensity[key], bins)
#        plt.hist(intensity[key], bins, alpha=0.5, label ='cluster %d' %key )
#    plt.legend(loc='upper right')
#    plt.xlabel('pressure difference [psi]')
#    plt.ylabel('number of occurence')
#    plt.show()


#plot S-N
    fig =plt.figure(figsize=(6,6), dpi=800, facecolor='w', edgecolor='k')
    bins = np.linspace(8, 80, 36)
    bin_aveg = np.zeros(len(bins)-1)
    markers = ['^','*','o', '>','+']
    for i in range(len(bins)-1):
        bin_aveg[i]= (bins[i]+bins[i+1])/2

    for key in sorted(assignment.keys()):
        intensity[key] = []
        for k in assignment[key]:
                intensity[key].append(max(data[k])-min(data[k]))

        tmp,base= np.histogram(intensity[key], bins)
        bin_nozero = []
        hist =[]
        for i in range(len(bin_aveg)):
            if tmp[i] != 0:
                bin_nozero.append(bin_aveg[i])
                hist.append(tmp[i])
        plt.scatter(hist, bin_nozero, s=50, marker=markers[key], label ='cluster %d' %key )
    leg=plt.legend(loc='best', fontsize=10)
    leg.get_frame().set_linewidth(0.0)
    plt.ylabel('pressure difference [psi]', fontsize=10)
    plt.xlabel('number of occurence', fontsize=10)
    plt.show()
    fig.savefig('SN.pdf', format='pdf', dpi=800)


    plt.style.use('grayscale')
    line = ['-','--','-','--']
    c= ['0','0','0.75','0.75']
#plot cdf
    ylim = 0
    fig =plt.figure(figsize=(4, 4), dpi=400, facecolor='w', edgecolor='k')
    bins = np.linspace(8, 80, 36)
    for key in sorted(assignment.keys()):
        intensity[key] = []
        if len(assignment[key]) > ylim:
            ylim = len(assignment[key])
        for k in assignment[key]:
            intensity[key].append(max(data[k])-min(data[k]))


        tmp,base= np.histogram(intensity[key], bins)
        #evaluate the cumulative
        cumulative = np.cumsum(tmp)
        # plot the cumulative function
        plt.step(base[:-1], cumulative, linestyle=line[key],color=c[key],
                 linewidth=3.0, label ='cluster %d' %key)

    leg=plt.legend(loc='best')
    leg.get_frame().set_linewidth(0.0)
    plt.xlabel('pressure difference [psi]')
    plt.ylabel('number of occurence')
    plt.xticks(range(10,90,10))
    plt.ylim([0, ylim+1])
    plt.show()
    fig.savefig('cdf.pdf', format='pdf', dpi=800)

#plot normalized cdf
    fig =plt.figure(figsize=(4, 4), dpi=400, facecolor='w', edgecolor='k')
    bins = np.linspace(8, 80, 36)
    for key in sorted(assignment.keys()):
        intensity[key] = []
        for k in assignment[key]:
                intensity[key].append(max(data[k])-min(data[k]))

        tmp,base= np.histogram(intensity[key], bins)

        #evaluate the cumulative
        cumulative = np.cumsum(tmp)/len(assignment[key])
        # plot the cumulative function
        plt.step(base[:-1], cumulative, linestyle=line[key],color=c[key],
                 linewidth=3.0, label ='cluster %d' %key)

    leg=plt.legend(loc='best')
    leg.get_frame().set_linewidth(0.0)
    plt.xlabel('Transient intensity [psi]')
    plt.ylabel('Probability')
    plt.xticks(range(10,90,10))
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()
    fig.savefig('cdf_norm.pdf', format='pdf', dpi=800)



    flat_intensity = [item for sublist in intensity.values() for item in sublist]
    plt.hist(flat_intensity, bins, alpha=1 )
    return intensity

#%%

def SumSquaredError(data, centroid, num_clust, assignment,w ) :
    SSE = np.zeros(num_clust)
    SSE_sum = 0
    for key in assignment :
        for k in assignment[key] :
            d = DTWDistance(data[k], centroid[key], w)
            SSE[key] += d**2
        SSE_sum += SSE[key]
    print ('Sum of Squared Error with %s cluster = %s' %(num_clust, SSE))

    return SSE_sum

#%%
def silhouette(dist, projct2d, num_clust, cluster_labels, cluster_alog ):

    plt.style.use('default')
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    plt.tight_layout()
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The ( num_clust+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(dist) + (num_clust + 1) * 10])


    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dist, cluster_labels, metric = "precomputed")
    print("For  num_clust =",num_clust,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dist, cluster_labels, metric = "precomputed")
    y_lower = 10

    for i in range(num_clust):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i



        color = cm.nipy_spectral(float(i+1)/ (num_clust+1))
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

#    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_title('(a)')
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    # 2nd Plot showing the actual clusters formed
    cluster_plot = cluster_labels+1
    colors = cm.nipy_spectral(cluster_plot.astype(float) /( num_clust+1))
    ax2.scatter(projct2d[:, 0], projct2d[:, 1], marker='o', s=50, lw=0, alpha=0.7,
                c=colors, edgecolor='k')



#    ax2.set_title("The visualization of the clustered data.")
    ax2.set_title('(b)')
    ax2.set_xlabel("Feature space ")
    ax2.set_ylabel("Feature space ")
#    plt.figtext(0.5,0.01, ("The average silhouette_score is : %.3g"  %silhouette_avg ),
#                           ha="center",  va="top", fontsize=14)
#    plt.suptitle(("Silhouette analysis for  %s clustering ""with  num_clust = %d "
#                  %(cluster_alog, num_clust)),fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return silhouette_avg
#%%

def k_means(data, projct2d, dist, num_clust, max_iter, randm_init, num_data = 25, w=10, tol = 1e-4,
                  plot = True, sil = True, SSE = True, sort = True, ch =True, method = 'K-Means'):
    min_cost = float('inf')
    plt.style.use('default')
    # random initialize 100 times and pick the best results to avoid local optimal
    for random_initial in range ( randm_init):
#        print("="*30)
#        print('random initialization # ', random_initial)

        center_shift_total = float('inf')
        centroids=random.sample(list(data),num_clust)
        counter=0

        while center_shift_total > tol and counter <= max_iter :

#            print('counter', counter)

            centroids_old = np.copy(centroids)

            counter +=1
            assignments={}
            #assign data points to clusters
            for ind,i in enumerate(data):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(centroids):
                    if LB_Keogh(i,j,w) < min_dist:
                        cur_dist=DTWDistance(i,j,w)
                        if cur_dist < min_dist:
                            min_dist=cur_dist
                            closest_clust = c_ind
                if closest_clust in assignments:
                    assignments[closest_clust].append(ind)
                else:
                    assignments[closest_clust]=[]
                    assignments[closest_clust].append(ind)

            # if there is no data assigned to a cluster, delete that cluster
            assignments = {k: v for k, v in assignments.items() if v}

            #recalculate centroids of clusters
            # 1--Clustering prototype : the average sequence
            for key in assignments:
                clust_sum=0
                for k in assignments[key]:
                    if len(data[k]) == num_data:
                        clust_sum=clust_sum+data[k]
                centroids[key]=[m/len(assignments[key]) for m in clust_sum]

            center_shift_total = \
                np.linalg.norm(np.subtract(centroids, centroids_old))
#            print ('shift', center_shift_total)
            if counter-1 == max_iter:
                print("Maximum number of iterations have been achieved...")
#            # 2--Clustering prototype : the medoid sequence
#            for key in assignments:
#                min_clust_sum = float('inf')
#                for i in assignments[key]:
#                    clust_sum=0
#                    for j in assignments[key]:
#                        clust_sum += DTWDistance(data[i], data[j], w)
#
#                    if clust_sum <min_clust_sum :
#                        centroids[key] = data[i]


        cost = cal_cost(data, assignments, centroids, w )

#        print('cost', cost)
        if cost < min_cost :
            min_cost = cost
            centroid = centroids
            assignment = assignments

    # prototype
#    centroid_orig = np.copy(centroid)
    for key in assignment:
        min_clust_sum = float('inf')
        for i in assignment[key]:
            clust_sum=0
            for j in assignment[key]:
                clust_sum += DTWDistance(data[i], data[j], w)
            if clust_sum <min_clust_sum :
                min_clust_sum = clust_sum
                centroid[key] = data[i]
#                centroid_orig[key] = data_orig[i]

    cluster_labels = np.zeros(len(data))

    for key in assignment :
        cluster_labels[assignment[key]] = key

#    intensity = transient_intensity(orig_data, assignment)

    if plot:
#        plot_cluster_k(data_orig, centroid_orig, assignment, num_clust,method)
        plot_cluster_k(data, centroid, assignment, num_clust,method)
    if ch :
        ch_score = calinski_harabaz_score(data, cluster_labels)
    if sil :
        sil_avg = silhouette(dist, projct2d,num_clust, cluster_labels, method)

    if SSE :
        SSE_sum = SumSquaredError(data, centroid, num_clust, assignment,w )

    if sort :
        sorted_data = sort_cluster(data,assignment)
    return centroid, assignment,  min_cost, sorted_data, sil_avg, SSE_sum, ch_score



#%%
def k_medoids(data, projct2d, dist, data_orig, num_clust, max_iter, randm_iter, w=10, tol = 1e-4,
                  plot = True, sil = True, ch = True, SSE= True, method = 'K-Medoids'):
    plt.style.use('default')
    min_cost = float('inf')
    # random initialize 100 times and pick the best results to avoid local optimal
    for random_initial in range ( randm_iter):
#        print("="*30)
#        print('random initialization # ', random_initial)

        center_shift_total = float('inf')
        centroids=random.sample(list(data),num_clust)
        centroids_orig=random.sample(list(data),num_clust)

        counter=0
        while center_shift_total > tol and counter <= max_iter :

#            print('counter', counter)

            centroids_old = np.copy(centroids)
            counter+=1
            assignments={}
            #assign data points to clusters
            for ind,i in enumerate(data):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(centroids):
                    if LB_Keogh(i,j,w) < min_dist:
                        cur_dist=DTWDistance(i,j,w)
                        if cur_dist < min_dist:
                            min_dist=cur_dist
                            closest_clust = c_ind
                if closest_clust in assignments:
                    assignments[closest_clust].append(ind)
                else:
                    assignments[closest_clust]=[]
                    assignments[closest_clust].append(ind)

            # if there is no data assigned to a cluster, delete that cluster
            assignments = {k: v for k, v in assignments.items() if v}

            #recalculate centroids of clusters
#            # 1--Clustering prototype : the average sequence
#            for key in assignments:
#                clust_sum=0
#                for k in assignments[key]:
#                    clust_sum=clust_sum+data[k]
#                centroids[key]=[m/len(assignments[key]) for m in clust_sum]

            # 2--Clustering prototype : the medoid sequence
            for key in assignments:
                min_clust_sum = float('inf')
                for i in assignments[key]:
                    clust_sum=0
                    for j in assignments[key]:
                        clust_sum += DTWDistance(data[i], data[j], w)

                    if clust_sum <min_clust_sum :
                        centroids[key] = data[i]
                        centroids_orig[key] = data_orig[i]

            center_shift_total = \
                np.linalg.norm(np.subtract(centroids, centroids_old))
#            print ('shift', center_shift_total)
#            if counter - 1== max_iter:
#                print("Maximum number of iterations have been achieved...")

        cost = cal_cost (data, assignments, centroids, w )
#        print('cost', cost)
        if cost < min_cost :
            min_cost = cost
#            centroid_orig = centroids_orig
            centroid = centroids
            assignment = assignments

    cluster_labels = np.zeros(len(data))
    for key in assignment :
        cluster_labels[assignment[key]] = key

#    centroid_orig = np.array(orig_data)[centroids_index]
    if plot:
#        plot_cluster_k(data_orig, centroid_orig, assignment, num_clust, method)
        plot_cluster_k(data, centroid, assignment, num_clust, method)

    if ch :
        ch_score = calinski_harabaz_score(data, cluster_labels)
    if sil :
        sil_avg = silhouette(dist,projct2d, num_clust, cluster_labels, method)

    if SSE :
        SSE_sum = SumSquaredError(data, centroid, num_clust, assignment,w )


    return centroid, assignment,  min_cost, sil_avg, SSE_sum, ch_score
#%%
def agglomerative(data, projct2d,dist, w, method,  num_clust, show_dend,
                  plot = True, sil = True, SSE = True,sort = True,ch =True  ) :

    plt.style.use('default')
    # show dendrograms
    # dendrogram can help determine the appropriate number of cluster.
    if show_dend :
        condense_dist = squareform(dist)
        linked = linkage(condense_dist, method)

        plt.figure(figsize=(8, 8))
        dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
        plt.title("Agglomeretic clustering with %s linkage" % method,  fontsize=14 )
        plt.show()


    cluster = AgglomerativeClustering( num_clust, affinity='precomputed', linkage='complete')
    cluster.fit_predict(dist)

 #assign clusters
    assignment={}
    for ind, cluster_id in enumerate(cluster.labels_):
        if cluster_id in assignment:
            assignment[cluster_id].append(ind)
        else:
            assignment[cluster_id]=[]
            assignment[cluster_id].append(ind)

            # if there is no data assigned to a cluster, delete that cluster
            assignment = {k: v for k, v in assignment.items() if v}

# prototype
    centroid = np.zeros(( num_clust, len(data[0])))
#    centroid_orig = np.copy(centroid)
    for key in assignment:
        min_clust_sum = float('inf')
        for i in assignment[key]:
            clust_sum=0
            for j in assignment[key]:
                clust_sum += DTWDistance(data[i], data[j], w)
            if clust_sum <min_clust_sum :
                min_clust_sum = clust_sum
                centroid[key] = data[i]
#                centroid_orig[key] = data_orig[i]

    if plot :
        plot_cluster_k(data, centroid, assignment, num_clust, method)
#        plot_cluster_hiera(data, assignment,  num_clust  )

    if ch :
        ch_score = calinski_harabaz_score(data, cluster.labels_)
    if sil :
        for ind_i,i in enumerate(data):
            for ind_j,j in enumerate(data):
                dist[ind_i,ind_j] = DTWDistance(i,j,w)
        sil_avg = silhouette(dist, projct2d, num_clust, cluster.labels_, 'Agglomerative (bottom-up)')

    if SSE :
        SSE_sum = SumSquaredError(data, centroid,  num_clust, assignment,w )

    return sil_avg, SSE_sum, ch_score



#%%
def search_find_density_peaks(data,projct2d, dist, w,percent,  plot = True, sil = True, SSE = True,sort = True,
                               ch =True, method = 'search_find_density_peaks'):
    plt.style.use('default')
    N = len(data)
    rho = np.zeros(N)
    delta = np.zeros(N)
    nneigh = np.zeros(N)
    gamma = np.zeros(N)
    ind = np.zeros(N)
    halo = np.zeros(N)

    assignment = {}

#    percent = 5.0
    print ('average percentage of neighbours (hard coded):', percent)

    position=round(N*N*percent/100)-1

    sda=np.sort(dist.flatten())
    dc=sda[position]

    print ('Computing Rho with gaussian kernel of radius: ', dc)

    rho = np.zeros(N)
    delta = np.zeros(N)
    nneigh = np.zeros(N)
    gamma = np.zeros(N)
    ind = np.zeros(N)
    halo = np.zeros(N)
#    # Guassian kernel
#    for i in range(0,N-1):
#        for j in range(i+1,N):
#            rho[i] = rho[i] + np.exp(- (dist[i,j]/dc)*(dist[i,j]/dc))
#            rho[j] = rho[j] + np.exp(- (dist[i,j]/dc)*(dist[i,j]/dc))

    # cutoff kernel
    for i in range(N-1):
        for j in range(i,N):
            if (dist[i,j]<dc):
                rho[i] += 1
                rho[j] += 1


    #standalize
    rho = rho/N

    maxd = np.amax(dist)

    ordrho = np.argsort(rho)[::-1]
#    rho_sorted = np.sort(rho)[::-1]

    delta[ordrho[0]] = -1
    nneigh[ordrho[0]] = 0

    #threshold
    rhomin = percent/100
    deltamin = 0.2

    for ii in range(1,N):
        delta[ordrho[ii]] = maxd
        for jj in range(0,ii):
            if dist[ordrho[ii],ordrho[jj]] < delta[ordrho[ii]]:
                delta[ordrho[ii]] = dist[ordrho[ii],ordrho[jj]]
                nneigh[ordrho[ii]] = int(ordrho[jj])

    delta[ordrho[0]]=np.amax(delta)
    #standalize
    delta = delta/delta[ordrho[0]]

    fig= plt.figure(figsize=(4,4), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(rho,delta, marker='o', c="k", alpha=1, edgecolor='k')
    plt.axhline(deltamin, color='r',linestyle ='--')
    plt.axvline(rhomin, color='r',linestyle ='--')
#    plt.scatter(rho,delta, marker='o', c="white", alpha=1, s=200, edgecolor='k')

#    for i in range(len(data)):
#        plt.scatter(rho[i],delta[i], marker='$%d$' % i, alpha=1,
#                    s=50, edgecolor='k')
#
    plt.xlabel(r"$\rho$ / total number of transients")
    plt.ylabel(r"$\delta$ / max($\delta)$")
    plt.show()
    #save
    fig.savefig('DecisionGraph.pdf', format='png', dpi=600)

    for i in range(N):
        ind[i]=i
        gamma[i] = rho[i]*delta[i]


    NCLUST = 0

    cl = -np.ones(N)
    icl =[]
    for i in range(N):
        if rho[i]>rhomin and delta[i] >deltamin:
            NCLUST = NCLUST+1
            cl[i] = NCLUST-1
            icl.append(i)
    icl = np.array(icl).astype(int)
    centroid = data[icl]
    #performing assignment
    for i in range(N):
        if cl[ordrho[i]] == -1 :
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    #halo
    for i in range(N):
        halo[i] = cl[i]

    if NCLUST>0 :
        bord_rho = np.zeros(NCLUST)
        for i in range(N-1):
            for j in range(i+1,N):
                if cl[i] != cl[j] and dist[i,j] <=dc:
                    rho_aver = (rho[i] + rho[j])/2
                    if rho_aver > bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver > bord_rho[int(cl[j])]:
                        bord_rho[int(cl[j])] = rho_aver

        for i in range(N):
            if rho[i] < bord_rho[int(cl[i])] :
                halo[i] = -1

    for i in range(NCLUST) :
        nc = 0
        nh = 0
        for j in range(N) :
            if cl[j] == i :
                nc +=1
                if i in assignment:
                    assignment[i].append(j)
                else:
                    assignment[i]=[]
                    assignment[i].append(j)

            if halo[j] == i :
                nh +=1

        print ('cluster: %d; center: %d; elements: %d; core: %d; HALO: %d'
               %(i,icl[i],nc,nh,nc-nh) )


    if plot :
        plot_cluster_k(data, centroid, assignment, NCLUST, method )
    if ch :
        ch_score = calinski_harabaz_score(data, cl)
    if sil :
        sil_avg = silhouette(dist, projct2d, NCLUST, cl, method)
    if SSE :
        SSE_sum = SumSquaredError(data, centroid,  NCLUST, assignment,w )
    return centroid, rho, delta, cl, sil_avg, SSE_sum, ch_score

#%%  Compute Affinity Propagation
def affinity_propagation(data, projct2d, dist, preference,  w,
                         plot = True, sil = True, SSE = True, ch = True,
                         method='Affinity Propagation' ):
    plt.style.use('default')
    af = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15,
                             copy=True, preference=preference, affinity='precomputed', verbose=False).fit(dist)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    num_clust = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % num_clust)

#    print("Silhouette Coefficient: %0.3f"
#          % metrics.silhouette_score(projct2d, labels, metric='euclidean'))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(num_clust), colors):
        class_members = labels == k
        cluster_center = projct2d[cluster_centers_indices[k]]
        plt.plot(projct2d[class_members, 0], projct2d[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in projct2d[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Estimated number of clusters: %d' % num_clust)
    plt.show()

#assignment
    assignment ={}
    for i in range( num_clust) :
        for j in range(len(data)) :
            if labels[j] == i :
                if i in assignment:
                    assignment[i].append(j)
                else:
                    assignment[i]=[]
                    assignment[i].append(j)
# prototype
    centroid = np.zeros(( num_clust, len(data[0])))
#    centroid_orig = np.copy(centroid)
    for key in assignment:
        min_clust_sum = float('inf')
        for i in assignment[key]:
            clust_sum=0
            for j in assignment[key]:
                clust_sum += DTWDistance(data[i], data[j], w)
            if clust_sum <min_clust_sum :
                min_clust_sum = clust_sum
                centroid[key] = data[i]
#                centroid_orig[key] = data_orig[i]

    if plot :
        plot_cluster_k(data, centroid, assignment, num_clust, method )

    if sil :
        sil_avg = silhouette(dist, projct2d, num_clust, labels, method)
    if ch :
        ch_score = calinski_harabaz_score(data, labels)
    if SSE :
        SSE_sum = SumSquaredError(data, centroid, num_clust, assignment,w )
    # Plot result
    plt.close('all')
    plt.figure(1)
    plt.clf()
    return sil_avg, ch_score, SSE_sum

#%%# Compute DBSCAN
def DBSCAN_clustering (data, projct2d, dist, w, eps, min_samples,
                         plot = True, sil = True, SSE = True, ch = True,
                         method='DBSCAN' ) :
    plt.style.use('default')
    db = DBSCAN(eps, min_samples).fit(projct2d)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    num_clust = len(set(labels)) - (1 if -1 in labels else 0)


    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = projct2d[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = projct2d[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % num_clust)
    plt.show()

    #assignment
    assignment ={}
    for i in range( num_clust) :
        for j in range(len(data)) :
            if labels[j] == i :
                if i in assignment:
                    assignment[i].append(j)
                else:
                    assignment[i]=[]
                    assignment[i].append(j)
# prototype
    centroid = np.zeros(( num_clust, len(data[0])))
#    centroid_orig = np.copy(centroid)
    for key in assignment:
        min_clust_sum = float('inf')
        for i in assignment[key]:
            clust_sum=0
            for j in assignment[key]:
                clust_sum += DTWDistance(data[i], data[j], w)
            if clust_sum <min_clust_sum :
                min_clust_sum = clust_sum
                centroid[key] = data[i]
#                centroid_orig[key] = data_orig[i]

    if plot :
        plot_cluster_k(data, centroid, assignment, num_clust, method )

    if sil :
        sil_avg = silhouette(dist, projct2d, num_clust, labels, method)
    if ch :
        ch_score = calinski_harabaz_score(data, labels)
    if SSE :
        SSE_sum = SumSquaredError(data, centroid, num_clust, assignment,w )


    return centroid, sil_avg, ch_score, SSE_sum
#%%
# def HDBSCAN_clustering (data, projct2d, dist, w, min_samples,
#                          plot = True, sil = True, SSE = True, ch = True,
#                          method='HDBSCAN' ) :
#     plt.style.use('default')
#     clusterer = hdbscan.HDBSCAN( min_samples).fit(projct2d)
#     labels = clusterer.labels_

# # Number of clusters in labels, ignoring noise if present.
#     num_clust = len(set(labels)) - (1 if -1 in labels else 0)

#     color_palette = sns.color_palette('deep', 8)
#     cluster_colors = [color_palette[x] if x >= 0
#                       else (0.5, 0.5, 0.5)
#                       for x in clusterer.labels_]
#     cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                              zip(cluster_colors, clusterer.probabilities_)]

#     plt.scatter(*projct2d.T, s=50, linewidth=0, c=cluster_member_colors)

#     #assignment
#     assignment ={}
#     for i in range( num_clust) :
#         for j in range(len(data)) :
#             if labels[j] == i :
#                 if i in assignment:
#                     assignment[i].append(j)
#                 else:
#                     assignment[i]=[]
#                     assignment[i].append(j)
# # prototype
#     centroid = np.zeros(( num_clust, len(data[0])))
# #    centroid_orig = np.copy(centroid)
#     for key in assignment:
#         min_clust_sum = float('inf')
#         for i in assignment[key]:
#             clust_sum=0
#             for j in assignment[key]:
#                 clust_sum += DTWDistance(data[i], data[j], w)
#             if clust_sum <min_clust_sum :
#                 min_clust_sum = clust_sum
#                 centroid[key] = data[i]
# #                centroid_orig[key] = data_orig[i]

#     if plot :
#         plot_cluster_k(data, centroid, assignment, num_clust, method )

#     if sil :
#         sil_avg = silhouette(dist, projct2d, num_clust, labels, method)
#     if ch :
#         ch_score = calinski_harabaz_score(data, labels)
#     if SSE :
#         SSE_sum = SumSquaredError(data, centroid, num_clust, assignment,w )


#     return sil_avg, ch_score, SSE_sum