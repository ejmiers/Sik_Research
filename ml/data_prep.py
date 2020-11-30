import numpy as np
from sklearn.preprocessing import StandardScaler
import os

PATH = "/media/ericmiers/Grad School Data/Research/Data/Simulated Signals/"
DEVICES = ["dev_0", "dev_1", "dev_2", "dev_3"]

def prepData(noise, dataset):

    path = PATH + dataset

    deviceSamples = []
    deviceLabels = []

    for device in DEVICES:
        devicePath = os.path.join(path, device)
        label = DEVICES.index(device)

        for signal in os.listdir(devicePath):
            signalPath = os.path.join(devicePath, signal)
            signalFile = next(s for s in os.listdir(signalPath) if s.endswith(str(noise) + ".data"))
            print("Loading {}".format(signalPath + "\\" + signalFile))
            signalData = np.fromfile(os.path.join(signalPath, signalFile), dtype=np.complex64)

            # Grab ~1000000 random sample from the data
            numSamples = 1000192
            numInputs = 128
            signalSamples = np.random.choice(signalData, numSamples)
            
            # Separate into real and imaginary components -> normalize for Gradient Descent
            real = signalSamples.real
            imag = signalSamples.imag

            #real = signalSamples.real / max(signalSamples.real)
            #imag = signalSamples.imag / max(signalSamples.imag)

            # Produce array: [[[I*128],[Q*128]], [[I*128],[Q*128]]...]
            iq_components = np.ravel(np.column_stack((real, imag))).reshape(-1, 2, 128)
            deviceSamples.append(iq_components)

            # Append Device Labels
            deviceLabels.append([label] * iq_components.shape[0])

    # Merge data, labels
    data = np.concatenate([samples for samples in deviceSamples])
    labels = np.concatenate([labels for labels in deviceLabels])
    print("Final Shape - {}: {}".format(dataset, data.shape))

    return data, labels


def normalize(trainData, testData):
    lenTrain = trainData.shape[0]
    lenTest = testData.shape[0]

    fullSet = np.concatenate((trainData, testData))

    # Normalize for Gradient Descent
    fullSet = fullSet.reshape(-1,2)
    scaler = StandardScaler() 
    fullSet = scaler.fit_transform(fullSet)
    fullSet = trainData.reshape(-1, 2, 128)

    # Re-split into training and testing
    trainDataNorm = fullSet[:lenTrain]
    testDataNorm = fullSet[len(fullSet)-lenTest:]

    # Check that the sets are the right sizes and return
    if (lenTrain == trainDataNorm.shape[0]) and (lenTest == testDataNorm.shape[0]):
        return trainDataNorm, testDataNorm
    
    print("ERROR: Training and Testing Dataset Dimensions Altered")


noise = 0.01 
trainData, trainLabels = prepData(noise, "train")
testData, testLabels = prepData(noise, "test")

trainDataNorm, testDataNorm = normalize(trainData, testData)

np.save("{}{}/samples_{}.npy".format(PATH, "train", noise), trainDataNorm)
np.save("{}{}/labels_{}.npy".format(PATH, "train", noise), trainLabels)

np.save("{}{}/samples_{}.npy".format(PATH, "test", noise), testDataNorm)
np.save("{}{}/labels_{}.npy".format(PATH, "test", noise), testLabels)