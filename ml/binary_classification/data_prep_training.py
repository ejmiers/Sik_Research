# Used to create a training and testing dataset from raw IQ data for a specified SNR
# Takes a subset of samples from each data file to account for memory constraints
# Data is normalized with the "StandardScaler" transformer -> mean of 0, stddev of 1 (Useful for Gradient Descent)
# Normalization model is saved to a file to be used to transform novel prediction data
# Saves training data and labels, testing data and labels as npy files
#
# Eric Miers
# Christopher Newport University
# January 2, 2021

import numpy as np
import os

def saveToFile(setType, dataType, device, data):
    np.save("{}\\binary_{}_{}_{}_{}_{}.npy".format(DATASET_PATH, setType, dataType, KNOWN_DEVICE, device, SNR), data)


def prepSamples(signalData, startIndex, stopIndex):
    signalSamples = signalData[startIndex:stopIndex]
    
    # Separate into real and imaginary components
    real = signalSamples.real
    imag = signalSamples.imag

    # Produce array: [[[I*128],[Q*128]], [[I*128],[Q*128]]...]
    iq_components = np.ravel(np.column_stack((real, imag))).reshape(-1, 2, 128)

    return iq_components


def prepData(SNR):

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)

        print("\n" + device)
        if device == KNOWN_DEVICE:
            label = 0
        else:
            label = 1

        signalFiles =  [s for s in os.listdir(devicePath) if s.endswith("_" + str(SNR) + ".data")]
   
        if len(signalFiles) == 0:
            continue

        for signalFile in signalFiles:

            print("Loading {}".format(devicePath + "\\" + signalFile))
            signalData = np.fromfile(os.path.join(devicePath, signalFile), dtype=np.complex64)
            print("Total Samples: {}".format(len(signalData)))
            print("label: {}".format(label))
            numInputs = 128

            # Designate Signal Data usage for each set.
            numSamplesTrain = 15000064
            numSamplesVal= 20000000
            numSamplesTest = 40000000

            if device == KNOWN_DEVICE: 
                numSamplesTrain = 140000000

            print("Num Samples Train: {}".format(numSamplesTrain))
            print("Num Samples Validate: {}".format(numSamplesVal))
            print("Num Samples Test: {}".format(numSamplesTest))
            
            # Partition the signal Data into training, validation, and test set files (sequentially)
            startIndexTrain = 0
            stopIndexTrain = numSamplesTrain
            iq_samplesTrain = prepSamples(signalData, startIndexTrain, stopIndexTrain)
            iq_labelsTrain = np.array([label] * iq_samplesTrain.shape[0])
            saveToFile("training", "samples", device, iq_samplesTrain)
            saveToFile("training", "labels", device, iq_labelsTrain)

            # Clear some un-needed memory
            iq_samplesTrain = None
            iq_labelsTest = None
            del iq_samplesTrain
            del iq_labelsTrain


            startIndexVal = stopIndexTrain
            stopIndexVal = startIndexVal + numSamplesVal
            iq_samplesVal = prepSamples(signalData, startIndexVal, stopIndexVal)
            iq_labelsVal = np.array([label] * iq_samplesVal.shape[0])
            saveToFile("validation", "samples", device, iq_samplesVal)
            saveToFile("validation", "labels", device, iq_labelsVal)

            # Clear some un-needed memory
            iq_samplesVal = None
            iq_labelsVal = None
            del iq_samplesVal
            del iq_labelsVal


            startIndexTest = stopIndexVal
            stopIndexTest = startIndexTest + numSamplesTest
            iq_samplesTest = prepSamples(signalData, startIndexTest, stopIndexTest)
            iq_labelsTest = np.array([label] * iq_samplesTest.shape[0])
            saveToFile("testing", "samples", device, iq_samplesTest)
            saveToFile("testing", "labels", device, iq_labelsTest)

            # Clear some un-needed memory
            iq_samplesTest = None
            iq_labelsTest = None
            del iq_samplesTest
            del iq_labelsTest


DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1", "RFD900_111", "RFD900_112", "RFD900_113", "RFD900_114"]
ROGUE_DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1", "RFD900_112", "RFD900_113", "RFD900_114"]
KNOWN_DEVICE = "RFD900_111"

SNR = "no-noise" 

PATH = "F:\\Research\\Data\\Hardware Signals\\"
DATASET = "binary_{}_{}-rogues_{}".format(KNOWN_DEVICE, len(ROGUE_DEVICES), SNR)
DATASET_PATH = PATH + DATASET

# Create Necessary Directories
if not os.path.isdir(DATASET_PATH):
    os.mkdir(DATASET_PATH)

prepData(SNR)