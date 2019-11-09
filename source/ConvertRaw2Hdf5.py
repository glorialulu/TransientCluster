# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:01:35 2018

@author: lx2347
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:09:13 2018

@author: lx2347
"""
import os

#Lu's module
from  read_raw import read_raw


#########################################################################

#---- [USER INPUT] ----
# Please use absolute location and adjust accordingly.
save_dir = 'C:\\Users\\lx2347\\Documents\\DetectCluster\\sample_data\\hdf5\\'
root = 'C:\\Users\\lx2347\\Documents\\DetectCluster\\sample_data\\raw\\'
stn = 'austin_21'
raw_freq = 64
for file in sorted(os.listdir(root)):
    if os.path.isdir(os.path.join(root, file)) :
        for sensor in sorted( os.listdir(os.path.join(root, file))):
            if str(sensor) == stn:
                print('='*50)

                # convert raw data to hdf5
                print( "Converting Folfer", file, "......")
                folder = os.path.join(root, file,sensor)
                filename =read_raw(root, folder,file, save_dir, stn)






