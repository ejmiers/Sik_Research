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
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def prepData(SNR):

    deviceSamplesTrain = []
    deviceLabelsTrain = []
    deviceSamplesTest = []
    deviceLabelsTest = []

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)
        label = DEVICES.index(device)

        signalFile =  next((s for s in os.listdir(devicePath) if s.endswith("_" + str(SNR) + ".data")), None)
   
        if not signalFile:
            continue

        print("Loading {}".format(devicePath + "\\" + signalFile))
        signalData = np.fromfile(os.path.join(devicePath, signalFile), dtype=np.complex64)

        # Grab 10000000 sample from the data for training, 2560000 samples for testing (80/20 rule)
        numSamplesTrain = 40000000
        numSamplesTest = 10000000
        numInputs = 128

        #signalSamples = np.random.choice(signalData, numSamples)
        signalSamplesTrain = signalData[:numSamplesTrain]
        signalSamplesTest = signalData[numSamplesTrain:numSamplesTrain+numSamplesTest]
        
        # Separate into real and imaginary components
        realTrain = signalSamplesTrain.real
        imagTrain = signalSamplesTrain.imag

        realTest = signalSamplesTest.real
        imagTest = signalSamplesTest.imag

        # Produce array: [[[I*128],[Q*128]], [[I*128],[Q*128]]...]
        iq_componentsTrain = np.ravel(np.column_stack((realTrain, imagTrain))).reshape(-1, 2, 128)
        iq_componentsTest = np.ravel(np.column_stack((realTest, imagTest))).reshape(-1, 2, 128)

        deviceSamplesTrain.append(iq_componentsTrain)
        deviceSamplesTest.append(iq_componentsTest)

        # Append Device Labels
        deviceLabelsTrain.append([label] * iq_componentsTrain.shape[0])
        deviceLabelsTest.append([label] * iq_componentsTest.shape[0])

    # Merge data, labels
    dataTrain = np.concatenate([samples for samples in deviceSamplesTrain])
    labelsTrain = np.concatenate([labels for labels in deviceLabelsTrain])

    dataTest = np.concatenate([samples for samples in deviceSamplesTest])
    labelsTest = np.concatenate([labels for labels in deviceLabelsTest])

    print("Final Shape - Training: {}".format(dataTrain.shape))
    print("Final Shape - Testing: {}".format(dataTest.shape))

    return dataTrain, labelsTrain, dataTest, labelsTest


def normalize(dataTrain, dataTest):

    # Normalize for Gradient Descent
    dataTrain = dataTrain.reshape(-1,2)
    scaler = StandardScaler() 
    dataTrain = scaler.fit_transform(dataTrain)
    dataTrain = dataTrain.reshape(-1, 2, 128)

    dataTest = dataTest.reshape(-1,2)
    dataTest = scaler.transform(dataTest)
    dataTest = dataTest.reshape(-1, 2, 128)

    # Save normalization model so predicted value scaling matches the training data
    scalerModelPath = "{}multiclass_data_normalization_scaler.bin".format(DATASET_PATH)
    dump(scaler, scalerModelPath, compress=True)

    return dataTrain, dataTest


SNR = "40dB"

PATH = "F:\\Research\\Data\\Hardware Signals\\"
# DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1", "RFD900_111", "RFD900_112", "RFD900_113", "RFD900_114"]
DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1"]
DATASET = "multiradio_{}-devices_{}\\".format(len(DEVICES), SNR)
DATASET_PATH = PATH + DATASET
dataNormalize = True

# Create Necessary Directories
if not os.path.isdir(DATASET_PATH):
    os.mkdir(DATASET_PATH)


dataTrain, labelsTrain, dataTest, labelsTest = prepData(SNR)

if dataNormalize:
    dataTrain, dataTest = normalize(dataTrain, dataTest)

np.save("{}multiclass_training_samples_{}.npy".format(DATASET_PATH, SNR), dataTrain)
np.save("{}multiclass_training_labels_{}.npy".format(DATASET_PATH, SNR), labelsTrain)

np.save("{}multiclass_testing_samples_{}.npy".format(DATASET_PATH, SNR), dataTest)
np.save("{}multiclass_testing_labels_{}.npy".format(DATASET_PATH, SNR), labelsTest)