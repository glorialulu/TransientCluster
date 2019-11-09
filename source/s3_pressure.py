"""
Simple function that can be imported to turn an S3 file into a pressure output
Author: Michael Allen (Visenti) with edits from Janice Zhuang
Python 2.7
"""
"""
pressure coefficients:

100PSI: 0.0019531250 36.8040000000
200PSI: 0.0039062500 88.3040000000
300PSI: 0.0058593750 139.8040000000
"""

from array import array
import os
import time
import sys

# two arguments - pressure range and filename
# we compute: sample rate vs. file size
# we compute: pressure reading from sample
def s3_file(range, filepath):
    # create appropriate array type
    file_data = array('h')
    # read file
    f = open(filepath, 'rb')
    cur_bytes = os.path.getsize(filepath)
    file_data.fromfile(f, int(cur_bytes/2))
    f.close()
    # extract filename from path, for timestamp part to work
    filename = os.path.basename(filepath)
    # determine timestamp
    try:
        # timestamp is in milliseconds since the epoch
        ts = int(filename.split("/")[-1].split(".")[0])
        # convert to usec for convenience
        ts = ts/1000.0
    except:
        print ("Error timestamp")
        return None
    # estimate sample rate
    rate = 0
    if cur_bytes == 15360:
        rate = 256.0
    elif cur_bytes == 7680:
        rate = 128.0
    elif cur_bytes == 3840:
        rate = 64.0
    else:
        print ("Error srate")
        return None # invalid sample rate for pressure
    interval = 1.0/rate # time interval per sample
    # compute pressure readings and timestamps
    # loop through
    m = 0.0
    b = 0.0
    if range == 100:
        m,b = 0.0019531250,36.8040000000
    elif range == 200:
        m,b = 0.0039062500,88.3040000000
    elif range == 300:
        m,b = 0.0058593750,139.8040000000
    else:
        print ("Error pressure range")
        return None # invalid pressure sensor range

    data_list = []
    for sample in file_data:
        stamp = time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime(ts))+".%03d"%(int((ts - int(ts))*1000))
        # print to screen - commented out for now
        #sys.stdout.write("%s,%.2f\n"%(stamp, sample*m+b))
        # store in list, format (timestamp-UTC, unix time, pressure-psi)
        data_list.append((stamp,ts,sample*m+b))
        # update timestamp for next sample
        ts += interval

    return data_list

"""
# TODO: timezone coordination
if __name__ == "__main__":
    s3_file(200, sys.argv[1])
"""
