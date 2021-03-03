# Used to create a dataset from raw IQ data for a specified SNR
# Takes a subset of samples from each data file to account for memory constraints
# Data is normalized with the StandardScaler model developed in the training set preparation script 
# Saves data and labels as npy files
#
# Eric Miers
# Christopher Newport University
# January 4, 2021

import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import os

def prepData(SNR):

    deviceSamples = []
    deviceLabels = []

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)

        if device == KNOWN_DEVICE:
            label = 0
        else:
            label = 1

        signalFile =  next((s for s in os.listdir(devicePath) if s.endswith("_" + str(SNR) + ".data")), None)
   
        if not signalFile:
            continue

        print("Loading {}".format(devicePath + "\\" + signalFile))
        signalData = np.fromfile(os.path.join(devicePath, signalFile), dtype=np.complex64)

        # Grab 10000000 samples from the data for testing
        numSamplesTest = 10000000
        numInputs = 128

        signalSamples = signalData[len(signalData)-numSamplesTest:len(signalData)]
        
        # Separate into real and imaginary components
        real = signalSamples.real
        imag = signalSamples.imag

        # Produce array: [[[I*128],[Q*128]], [[I*128],[Q*128]]...]
        iq_components = np.ravel(np.column_stack((real, imag))).reshape(-1, 2, 128)

        deviceSamples.append(iq_components)

        # Append Device Labels
        deviceLabels.append([label] * iq_components.shape[0])

    # Merge data, labels
    data = np.concatenate([samples for samples in deviceSamples])
    labels = np.concatenate([labels for labels in deviceLabels])
    labels.reshape(-1, 1)

    print("Final Shape: {}".format(data.shape))

    return data, labels


def normalize(data):

    # Normalize for Gradient Descent
    data = data.reshape(-1,2)
    scaler = load("{}binary_data_normalization_scaler_{}_{}.bin".format(DATASET_PATH, KNOWN_DEVICE, NORMALIZATION_SCALER))
    data = scaler.transform(data)
    data = data.reshape(-1, 2, 128)

    return data


PATH = "F:\\Research\\Data\\Hardware Signals\\"
DATASET = "binary_mRo_1_8-rogues_no_noise\\"
DATASET_PATH = PATH + DATASET
DEVICES = ["mRo_1", "mRo_2"]
KNOWN_DEVICE = "mRo_1"
NORMALIZATION_SCALER = "mRo_2"
normalizeData = True

SNR = "2dB" 
data, labels = prepData(SNR)

if normalizeData:
    data = normalize(data)

np.save("{}binary_prediction_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, DEVICES[1], SNR), data)
np.save("{}binary_prediction_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, DEVICES[1], SNR), labels)