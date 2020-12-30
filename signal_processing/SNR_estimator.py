
import numpy as np
import os

# PATH = "/media/ericmiers/Grad School Data/Research/Data/Simulated Signals/"
# DEVICES = ["dev_0", "dev_1", "dev_2", "dev_3"]

PATH = "/media/ejmie518/Grad School Data/Research/Data/Hardware Signals/"
DEVICES = ["mRo_1"]

noiseAmp = 0.193

def calculateSNR(signal, noise):
    # Compute power of noise (dB)
    noiseMagnitude = np.absolute(noise)
    noiseAveragePower = np.mean(np.square(noiseMagnitude))
    noiseAveragePowerLog = 10 * np.log10(noiseAveragePower)

    # Compute power of signal (dB)
    signalMagnitude = np.absolute(signal)
    signalAveragePower  = np.mean(np.square(signalMagnitude))
    signalAveragePowerLog = 10 * np.log10(signalAveragePower)

    # Compute SNR
    SNR = signalAveragePowerLog - noiseAveragePowerLog

    #print(f"block SNR: {SNR} (dB)")

    return SNR


def getSNRs():

    signalSNRs = []

    for device in DEVICES:
        devicePath = os.path.join(PATH, device)

        signalFiles = [s for s in os.listdir(devicePath) if s.startswith("filtered") and s.endswith(str(noiseAmp) + ".data")]
        noiseFiles = [s for s in os.listdir(devicePath) if s.startswith("noise_reduced") and s.endswith(str(noiseAmp) + ".data")]

        for i in range(0, len(signalFiles)):
            # #signalPath = os.path.join(devicePath, signal)
            # signalFile = next((s for s in os.listdir(signalPath) if s.startswith("filtered") and s.endswith(str(noiseAmp) + ".data")), None)

            # if not signalFile:
            #     continue
            signalFile = signalFiles[i]
            noiseFile = noiseFiles[i]

            print("Loading {}".format(devicePath + "\\" + signalFile))
            signalData = np.fromfile(os.path.join(devicePath, signalFile), dtype=np.complex64)
            print("Loading {}".format(devicePath + "\\" + noiseFile))
            noiseData = np.fromfile(os.path.join(devicePath, noiseFile), dtype=np.complex64)

            blockSize = 128
            signalSNR = []

            print("Calculating Average SNR...")
            # Implements the sliding window algorithm
            while signalData.shape[0] > 0:

                # Check to see if the next block is cut off by the end of the array
                if signalData.shape[0] > blockSize:
                    blockEnd = blockSize
                else:
                    blockEnd = signalData.shape[0]
                
                signalBlock = signalData[0:blockEnd]
                noiseBlock = noiseData[0:blockEnd]
                signalSNR.append(calculateSNR(signalBlock, noiseBlock))

                # Re-slice array by removing the current block
                signalData = signalData[blockEnd:]
                noiseData = noiseData[blockEnd:]
            
            signalSNRMean = np.mean(signalSNR)
            print(f"Signal SNR Mean: {signalSNRMean} (dB)\n")

            signalSNRs.append(signalSNRMean)
        
    return signalSNRs


SNRs = getSNRs()

for SNR in SNRs:
    print(f"\nSignal SNR: {SNR} (db)")
