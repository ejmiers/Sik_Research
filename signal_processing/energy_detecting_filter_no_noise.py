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

def calculateThreshold(noiseDeviation, probabilityFA, n):
    return 10*np.square(noiseDeviation) * (abs(norm.ppf(probabilityFA)) * np.sqrt(2*n) + n)

def filterSignal(rawData, blockSize, threshold): #noiseData, blockSize, threshold):

    # Use a standard list for filtered data
    # Python list allows for faster re-allocation than numpy array
    # Memory overhead is sacrificed for time
    filteredSignal = []
    #reducedNoise = []
    blockNum = 0

    # Implements the sliding window algorithm
    while rawData.shape[0] > 0:

        # Check to see if the next block is cut off by the end of the array
        if rawData.shape[0] > blockSize:
            blockEnd = blockSize
        else:
            blockEnd = rawData.shape[0]
        
        # Perform FFt on block, convert to squared magnitude
        samples = rawData[0:blockEnd]
        #noise = noiseData[0:blockEnd]

        block = np.fft.fft(samples)
        blockMean = np.mean(np.square(np.absolute(block)))

        # Add the samples in the block to the filtered sample array if the block's mean is greater or equal to the threshold
        if blockMean >= threshold:
            #print("block {} above threshold".format(blockNum))
            #filteredData.extend(np.fft.ifft(block).astype(np.complex64).tolist())
            filteredSignal.extend(samples.astype(np.complex64).tolist())
            #reducedNoise.extend(noise.astype(np.complex64).tolist())
            
        # Re-slice array by removing the current block
        rawData = rawData[blockEnd:]
        #noiseData = noiseData[blockEnd:]
        blockNum += 1
    
    return filteredSignal #, reducedNoise


# Input Data Files
device = "3DR_TL1"
noiseAmplitude = "no_noise"

# dataSampleFile = "F:/Research/Data/Hardware Signals/{}/{}_{}.data".format(device, noiseSeed, noiseAmplitude)
# noiseSampleFile = "F:/Research/Data/Hardware Signals/{}/noise_{}_{}.data".format(device, noiseSeed, noiseAmplitude)
dataSampleFile = "F:/Research/Data/Hardware Signals/{}/{}.data".format(device, noiseAmplitude)

# Load the input files into numpy arrays and convert the array to real values 
rawData = np.fromfile(dataSampleFile, dtype=np.complex64)
#noiseData = np.fromfile(noiseSampleFile, dtype=np.complex64)
noiseData = []

if len(noiseData) > 0:
    print("Computing noise amplitude...")
    noiseDeviation = np.std(noiseData)
else:
    noiseDeviation = 0.001757

# Define the threshold and sample block size.
blockSize = 128
threshold = calculateThreshold(noiseDeviation, 0.000009, blockSize)

print("Noise Deviation: {}".format(noiseDeviation))
print("Energy Level Threshold: {}".format(threshold))

# Filter the Dataset to only include samples above the threshold
#filteredSignal, reducedNoise = filterSignal(rawData, noiseData, blockSize, threshold)
filteredSignal = filterSignal(rawData, blockSize, threshold)


# Output Data Files
#filteredSignalFile = "F:/Research/Data/Hardware Signals/{}/filtered_{:4f}_{}_{}.data".format(device, threshold, noiseSeed, noiseAmplitude)
filteredSignalFile = "F:/Research/Data/Hardware Signals/{}/filtered_{:4f}_{}.data".format(device, threshold, noiseAmplitude)

#reducedNoiseFile = "F:/Research/Data/Hardware Signals/{}/noise_reduced_{:4f}_{}_{}.data".format(device, threshold, noiseSeed, noiseAmplitude)

# Write the filtered data to the output file
np.array(filteredSignal).astype(np.complex64).tofile(filteredSignalFile)
#np.array(reducedNoise).astype(np.complex64).tofile(reducedNoiseFile)

# Log the number of samples that met the thresholdgit stat
print("")
print("Initial Number of Samples: {}".format(len(rawData)))
print("Final Number of samples: {}".format(len(filteredSignal)))
print("")