# Used to create a training and testing dataset from raw IQ data
# Takes a random subset of samples from each data file to account for memory constraints
# Data is normalized with the "StandardScaler" transformer -> mean of 0, stddev of 1 (Useful for Gradient Descent)
# Saves training data and labels, testing data and labels as npy files
#
# Eric Miers
# Christopher Newport University
# November 30, 2020

import numpy as np
from sklearn.preprocessing import StandardScaler
import os

PATH = "/media/ericmiers/Grad School Data/Research/Data/Simulated Signals/"
DEVICES = ["dev_0", "dev_1", "dev_2", "dev_3"]

def prepData(noise):

    deviceSamples = []
    deviceLabels = []

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)
        label = DEVICES.index(device)

        for signal in os.listdir(devicePath):
            signalPath = os.path.join(devicePath, signal)
            signalFile = next(s for s in os.listdir(signalPath) if s.endswith(str(noise) + ".data"))
            print("Loading {}".format(signalPath + "\\" + signalFile))
            signalData = np.fromfile(os.path.join(signalPath, signalFile), dtype=np.complex64)

            # Grab ~5000000 random sample from the data
            #numSamples = 5000064
            numSamples = 10000000
            numInputs = 128
            signalSamples = np.random.choice(signalData, numSamples)
            #signalSamples = signalData[:numSamples]
            
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
    scaler = StandardScaler() 
    data = scaler.fit_transform(data)
    data = data.reshape(-1, 2, 128)

    return data


noise = 0.01 
data, labels = prepData(noise)

data = normalize(data)

np.save("{}samples_{}.npy".format(PATH, noise), data)
np.save("{}labels_{}.npy".format(PATH, noise), labels)