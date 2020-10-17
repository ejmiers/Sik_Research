# Filters captured signal IQ samples with a specified engergy spectrum threshold.
# Samples are divided into blocks of size 2048. Blocks that have an average magnitude
# above the threshold are written to a new data file.
#
# Input File format: float32 complex binary data file
# Output File format: float32 complex binary data file
#
# Eric Miers
# Christopher Newport University
# October 17, 2020

import numpy as np

# Input and Output Files
dataFile = "../Data/Sik_Capture_Raw.data"
newFile = "../Data/Sik_Capture_Filtered_1e-3Thresh.data"

# Load the input file into numpy array and convert the array to real values 
rawData = np.fromfile(dataFile, dtype=np.complex64)
rawDataReal = rawData.real

# Delete the original array to clear memory, initialize numpy array to store samples above the threshold
del rawData
filteredData = np.array([])

# Define the threshold and sample block size.
threshold = 1e-3
blockSize = 2048

# Define values for sliding-window algorithm
arraySize = rawDataReal.shape[0]
blockStart = 0
blockEnd = 0
blockNum = 0

# Implements the sliding window algorithm
while blockStart < arraySize:

    # Check to see if the next block is cut off by the end of the array
    if arraySize - blockStart > blockSize:
        blockEnd = blockStart + blockSize
    else:
        blockEnd = arraySize
    
    block = rawDataReal[blockStart:blockEnd]

    # Add the samples in the block to the filtered sample array if the block's mean is greater or equal to the threshold
    if np.mean(block) >= threshold:
        print("block {} above threshold".format(blockNum))
        if (filteredData.shape[0] == 0):
            filteredData = block
        else:
            filteredData = np.append(filteredData, block)
    
    # Slide the window for the next block
    blockStart += blockSize
    blockNum += 1

# Clear the original data from memory, convert the filtered data back to complex
del rawDataReal
filteredData = filteredData.astype(complex)

# Write the filtered data to the output file
filteredData.astype('float32').tofile(newFile)

# Log the number of samples that met the threshold
print("")
print("Final Number of samples: {}".format(filteredData.shape[0]))
print("")