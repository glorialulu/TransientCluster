# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:16:47 2018

@author: lx2347
"""
# file reading
import pandas as pd
#number computation
import numpy as np
# plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def read_data_from_csv(filename):
    """Read data from .csv file.

    Parameters
    ----------
    filename : sting
        csv filename
    """
    df = pd.read_csv(filename,header=0,
                     dtype={"Timestamp_UTC": object,
                            "EpochTime": np.float64, "Pressure_psi":np.float64})
    df= pd.DataFrame(df)
    return df

def read_data_from_hdf5(filename) :
    """read data from .hdf5 file.

    Parameters
    ----------
    filename : string
        .hdf5 filename
    """
    df = pd.read_hdf(filename)
    return df

def plot_data (x, y, save_dir):
    """ plot the original data """

    fig= plt.figure(figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.plot_date(x, y, fmt='-',tz="US/Central",xdate=True)
    plt.xlabel('Time')
    plt.ylabel('Pressure (psi)')
    plt.title("Pressure Plot ")
    plt.show()
    #save
    fig.savefig(save_dir + 'OriginalPressure.png', format='png', dpi=600)


def preprocessing (filename, save_dir, plot = True):
    """ read the high frequency data from csv files
    Usage:   x, signal = preprocessing(file, save_dir, plot = False)
    """
    # Reading input file......
    if filename.split('.')[-1] == "h5" :
        df=read_data_from_hdf5(filename)
    elif filename.split('.')[-1] == "csv" :
        df=read_data_from_csv(filename)
    else :
        print ("ERROR: file format not supportted.")


    #Preparing the date and pressure data to desired format...
    # convert object to datatime format
    df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC'], format='%Y/%m/%d %H:%M:%S.%f')
    # convert UTC time to local time
    df['Timestamp_UTC'] = df['Timestamp_UTC'].dt.tz_localize("UTC")
    Localizetime =df['Timestamp_UTC'].dt.tz_convert("US/Central")
#    df['Timestamp_UTC'] = df['Timestamp_UTC']
    x=Localizetime#.as_matrix()

    # use matplotlib 2.2.2 or later version for this function.
    # convert datetime to number
    x = mdate.date2num(x)
    signal=df['Pressure_psi']#.as_matrix()

    # Printing the original pressure data ...
    if plot :
       plot_data (x, signal, save_dir)

    return x, signal


