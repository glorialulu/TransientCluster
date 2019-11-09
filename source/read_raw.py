"""
Reads RAW sensor-pressure files downloaded from S3 browser,
and converts into CSVs
Author: Janice Zhuang
Python 2.7
Edited by Lu Xing
Python 3.6

Some notes:
Time elapsed = approx. 6 minutes for 1 station in 1 day
s3_file function takes the longest (~4 min)
"""

import glob
import os
import pandas as pd

from s3_pressure import s3_file # function from Visenti to convert from RAW


def read_raw(root, folder,num, save_dir, stn, save_as='h5') :

    # What is my pressure range?
    # Austin, TX = 200 PSI
    pRange = 200

    #---- CONVERT DATA ----
    # Get RAW files
    # Reads all .raw files in given folder
    path = os.path.join(root, folder, '*.raw')
    files = glob.glob(path)
    print ( "Total number of .raw files" , len(files))

    # Loop through files and append data to list
    data_list = []
    for f in files :
        # Run Visenti's converter function
        # Returns list of all samples in PSI w/ time
        dataOutput = s3_file(pRange, f)
        # data_list = data_list + dataOutput # this is too slow
        #instead, use the following:
        data_list += dataOutput

#        i += 1
#        if(i%500 == 0):
#          toc = time.time()
#          convert_time = np.append(convert_time,toc-tic)
#          print ((float(i)/len(files))*100, "%", "Time spent" , toc-tic, "s")


    # Write list to dataframe
    df = pd.DataFrame(data_list)
    df.columns = ['Timestamp_UTC', 'EpochTime', 'Pressure_psi']


#---- WRITE TO FILE ----
    # create folder
    try:
        os.makedirs(save_dir + stn)
    except FileExistsError:
        # directory already exists
        pass
    # Write to a HDF5
    if save_as == 'h5' :
        filename = save_dir + stn +'\\' + stn  + '-'+ str(num)+'.h5'
        df.to_hdf(filename,'Table')
    # write to csv file
    elif save_as == 'csv':
        filename = save_dir + stn +'\\' + stn  + '-'+ str(num)+'.csv'
        df.to_hdf(filename,'Table')
    else :
        print ("save format not supported")
    return filename


