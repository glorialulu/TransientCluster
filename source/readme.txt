This folder contains:
1. ConvertRaw2Hdf5.py
	converts raw data files to hdf5.
	This function calls read_raw
2. main.py 
	detects transients in the raw data, and discovers the typical patterns of the transients.
	This main function calls:
	- preprocessing.py: reads the files and prepare the data in the appropriate format. 
	- segmentation.py: reduces the dimentionality (resolution) of the high resolution data.
	- detect_cusum.py: detects the pressure transients using modified CUSUM algorithm.
	- zoom.py: zooms into the time when transients happen.
	- clustering.py: discovers the typical patterns of pressure transients using clustering algorithms.