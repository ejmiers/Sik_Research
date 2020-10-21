# Filters captured signal IQ samples with a specified engergy spectrum threshold.
# Samples are divided into blocks of size 2048. Blocks that have an average squared magnitude
# above the threshold are written to a new data file.
#
# Input File format: float32 complex binary data file
# Output File format: float32 complex binary data file
#
# Eric Miers
# Christopher Newport University
# October 17, 2020

import numpy as np
from scipy.stats import norm

def calculateThreshold(noiseAmplitude, probabilityFA, n):
    return noiseAmplitude**2 * (abs(norm.ppf(probabilityFA)) * (2*n)**0.5 + n)


def filterData(rawData, blockSize, threshold):

    # Use a standard list for filtered data
    # Python list allows for faster re-allocation than numpy array
    # Memory overhead is sacrificed for time
    filteredData = []
    blockNum = 0

    # Implements the sliding window algorithm
    while rawData.shape[0] > 0:

        # Check to see if the next block is cut off by the end of the array
        if rawData.shape[0] > blockSize:
            blockEnd = blockSize
        else:
            blockEnd = rawData.shape[0]
        
        # Perform FFt on block, convert to squared magnitude
        block = np.fft.fft(rawData[0:blockEnd])
        blockMean = np.mean(np.square(np.absolute(block)))

        #print(blockMean)

        # Add the samples in the block to the filtered sample array if the block's mean is greater or equal to the threshold
        if blockMean >= threshold:
            print("block {} above threshold".format(blockNum))
            filteredData.extend(np.fft.ifft(block).astype('float32').tolist())

        # Re-slice array by removing the current block
        rawData = rawData[blockEnd:]
        blockNum += 1
    
    return filteredData



# Define the https://github.com/ejmiers/Sik_Research.git://github.com/ejmiers/Sik_Research.gitreshold and sample block size.
#threshold = 1e-3
blockSize = 128
threshold = calculateThreshold(1, 0.10, blockSize)
print("Energy Level Threshold: {}".format(threshold))

# Input and Output Data Files
dataFile = "../../Lab PC/Data/Sik_Capture_Raw.data"
newFile = "../../Lab PC/Data/Sik_Capture_Filtered_{:4f}.data".format(threshold)

# Load the input file into numpy array and convert the array to real values 
rawData = np.fromfile(dataFile, dtype=np.complex64)

# Filter the Dataset to only include samples above the threshold
filteredData = filterData(rawData, blockSize, threshold)

# Write the filtered data to the output file
np.array(filteredData).astype('float32').tofile(newFile)

# Log the number of samples that met the threshold
print("")
print("Energy Level Threshold: {}".format(threshold))
print("Final Number of samples: {}".format(len(filteredData)))
print("")