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

PATH = "F:\\Research\\Data\\Hardware Signals\\"
DEVICES = ["mRo_1", "mRo_2", "3DR_T1", "3DR_TL1", "RFD900_111", "RFD900_112", "RFD900_113", "RFD900_114"]
normalize = True

def prepData(noise):

    deviceSamples = []
    deviceLabels = []

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)
        label = DEVICES.index(device)

        signalFile =  next((s for s in os.listdir(devicePath) if s.endswith(str(noise) + ".data")), None)
   
        if not signalFile:
            continue

        print("Loading {}".format(devicePath + signalFile))
        signalData = np.fromfile(os.path.join(devicePath, signalFile), dtype=np.complex64)

        # Grab 10000000 random sample from the data for training, 2560000 samples for testing (80/20 rule)
        numSamplesTest = 10000000
        numInputs = 128

        #signalSamples = np.random.choice(signalData, numSamples)
        signalSamples = signalData[:numSamplesTrain]
        
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

    print("Final Shape: {}".format(data.shape))

    return data, labels


def normalize(data):

    # Normalize for Gradient Descent
    data = data.reshape(-1,2)
    scaler = load("{}multiclass_data_normalization_scaler.bin".format(PATH))
    data = scaler.transform(data)
    data = dataTrain.reshape(-1, 2, 128)

    return data


noise = "35dB" 
data, labels = prepData(noise)

if normalize:
    data = normalize(data)

np.save("{}multiclass_samples_{}.npy".format(PATH, noise), data)
np.save("{}multiclass_labels_{}.npy".format(PATH, noise), labels)