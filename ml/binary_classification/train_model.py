from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
from functools import partial
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def writeSummaryFile():
    filename = runPath + "\\training_summary.txt"

    with open(filename, "w") as f:
        f.write("SUMMARY\n")
        f.write("=======\n\n")

        f.write("Model Description\n")
        f.write("-----------------\n")
        f.write("Number Hidden Layers: {}\n".format(numHiddenLayers))
        f.write("Number Nodes Hidden Layer: {}\n".format(sizeHiddenLayer))
        f.write("Dropout Rate Hidden Layer: {}\n".format(dropoutRate))
        f.write("Activation Function Hidden Layer: {}\n".format(activationHidden))
        f.write("Kernel Initializer Hidden Layer: {}\n".format(kernelInitializer))
        #f.write("Kernel Regularizer: {}\n".format(kernelRegularizerString))
        f.write("Activation Function Output Layer: {}\n".format(activationOutput))
        f.write("Loss Function: {}\n".format(lossFunction))
        f.write("Optimizer: {}\n".format(optimizer))
        f.write("Learning Rate: {}\n".format(learningRate))
        f.write("Momentum: {}\n".format(optMomentum))
        f.write("Batch Size: {}\n\n".format(batchSize))

        f.write("Run Description\n")
        f.write("---------------\n")
        f.write("Known Device: {}\n".format(KNOWN_DEVICE))
        f.write("Number Cross-Validation Folds: {}\n".format(numFolds))
        f.write("Number Epochs: {}\n".format(numEpochs))
        f.write("Early Stopping Patience: {}\n".format(esPatience))
        f.write("Early Stopping Mointor: {}\n".format(esMonitor))
        f.write("Best Model Metric: {}\n\n".format(bestModelMetric))


def writeResultsToSummary(foldLoss, foldAccuracy, totalTime):
    filename = runPath + "\\training_summary.txt"

    with open(filename, "a") as f:
        f.write("Run Results\n")
        f.write("---------------\n")

        for i in range(0, numFolds):
            f.write("Fold {} - validation loss: {}, validation accuracy: {}\n".format(i+1, foldLoss[i], foldAccuracy[i]))
            
        f.write("\nAverage Model Loss: {}\n".format(np.mean(foldLoss)))
        f.write("Standard Deviation Model Loss: {}\n\n".format(np.std(foldLoss)))

        f.write("Average Model Accuracy: {}\n".format(np.mean(foldAccuracy)))
        f.write("Standard Deviation Model Accuracy: {}\n\n".format(np.std(foldAccuracy)))

        # f.write("Best Model Loss: {}\n".format(bestModelLoss))
        # f.write("Best Model Accuracy: {}\n\n".format(bestModelAccuracy))

        f.write("Total trainining time (s): {}".format(totalTime))


#====================================GLOBALS============================================================

ROGUE_DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1", "RFD900_112", "RFD900_113", "RFD900_114"]
KNOWN_DEVICE = "RFD900_111"

SNR = "no_noise"

PATH = "F:\\Research\\Data\\Hardware Signals\\"
DATASET_PATH = "F:\\Research\\Data\\Hardware Signals\\binary_RFD900_111_8-rogues_no_noise\\"

trainingDate = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
runPath = PATH + "models\\binary\\" + trainingDate + " ({} Known)".format(KNOWN_DEVICE)
bestModelPath = runPath + "\\best_model.h5"

# Create Necessary Directories
if not os.path.isdir(PATH + "models\\binary"):
    if not os.path.isdir(PATH + "models"):
        os.mkdir(PATH + "models")
    os.mkdir(PATH + "models\\binary")
os.mkdir(runPath)

#====================================MODEL PARAMETERIZATION=========================================================

numHiddenLayers = 3
sizeHiddenLayer = 300
dropoutRate = 0.5
batchSize = 128
numEpochs = 300
activationHidden = "elu"
kernelInitializer = "he_normal"
kernelRegularizer = keras.regularizers.l2(0.0001)
kernelRegularizerString = "L2 - 0.0001"
activationOutput = "sigmoid"
lossFunction = "binary_crossentropy"
optimizer = "SGD"
learningRate = 0.01
optMomentum = 0.9
esMonitor = "validation loss - minimum"
bestModelMetric = "validation accuracy - maximum"
esPatience = 12

numFolds = len(ROGUE_DEVICES)
foldAccuracy = []
foldLoss = []

#===============================TRAINING==============================================================

writeSummaryFile()
timeStart = time.time()

for fold in range(0, numFolds):
    holdoutDevice = ROGUE_DEVICES[fold]

    # Setup the training set
    X_train = np.load("{}binary_training_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))
    Y_train = np.load("{}binary_training_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))

    for rogue in ROGUE_DEVICES:
        if rogue != holdoutDevice:
            X_temp = np.load("{}binary_training_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, rogue, SNR))
            Y_temp = np.load("{}binary_training_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, rogue, SNR))

            X_train = np.append(X_train, X_temp, axis=0)
            Y_train = np.append(Y_train, Y_temp, axis=0)

    # Normalize data - save the normalizer to a file
    X_train = X_train.astype('float32')
    X_train = X_train.reshape(-1,2)
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 2, 128)


    scalerModelPath = "{}binary_data_normalization_scaler_{}_{}.bin".format(DATASET_PATH, KNOWN_DEVICE, holdoutDevice)
    dump(scaler, scalerModelPath, compress=True)

    # Load the validation set - formed from the heldout radio
    X_val_known = np.load("{}binary_validation_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))
    Y_val_known = np.load("{}binary_validation_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))
    X_val_rogue = np.load("{}binary_validation_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, holdoutDevice, SNR))
    Y_val_rogue = np.load("{}binary_validation_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, holdoutDevice, SNR))
    X_val = np.append(X_val_known, X_val_rogue, axis=0)
    Y_val = np.append(Y_val_known, Y_val_rogue, axis=0)

    # Clear some memory
    X_val_known = X_val_rogue = Y_val_known = Y_val_rogue = None
    del X_val_known, X_val_rogue, Y_val_known, Y_val_rogue

    # Normalize the data to the same scale as the training set
    X_val = X_val.astype('float32')
    X_val = X_val.reshape(-1,2)
    X_val = scaler.transform(X_val)
    X_val = X_val.reshape(-1, 2, 128)


    # Shuffle Data
    X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)

    print("Count known radios - train: {}".format(list(Y_train).count(0)))
    print("Count rogue radios - train: {}\n".format(list(Y_train).count(1)))

    print("Count known radios - validation: {}".format(list(Y_val).count(0)))
    print("Count rogue radios - validation: {}\n".format(list(Y_val).count(1)))

    # Build the DNN model
    RegularizedDense = partial(keras.layers.Dense,
                            activation=activationHidden,
                            kernel_initializer=kernelInitializer,
                            kernel_regularizer=kernelRegularizer 
                            )

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[2,128]),
        RegularizedDense(sizeHiddenLayer),
        keras.layers.Dropout(rate=dropoutRate),
        RegularizedDense(sizeHiddenLayer),
        keras.layers.Dropout(rate=dropoutRate),
        RegularizedDense(sizeHiddenLayer),
        keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        keras.layers.Dense(1, activation=activationOutput)
    ])

    model.compile(loss = lossFunction,
                optimizer = keras.optimizers.SGD(lr=learningRate, momentum=optMomentum),
                #optimizer = keras.optimizers.Nadam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                metrics=["accuracy"])

    model.summary()
    print('\n=============================================================')
    print(f'Training fold {fold+1}...\n')

    # Implement early stopping with checkpoints to curb overfitting
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esPatience)
    mc = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=numEpochs, batch_size=batchSize, callbacks=[es, mc])

    # Clear some memory now that training is done
    X_train = Y_train = X_val = Y_val = None
    del X_train, Y_train, X_val, Y_val

    # Load and normalize the test set - formed from the heldout radio
    X_test_known = np.load("{}binary_testing_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))
    Y_test_known = np.load("{}binary_testing_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, KNOWN_DEVICE, SNR))
    X_test_rogue = np.load("{}binary_testing_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, holdoutDevice, SNR))
    Y_test_rogue = np.load("{}binary_testing_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, holdoutDevice, SNR))
    X_test = np.append(X_test_known, X_test_rogue, axis=0)
    Y_test = np.append(Y_test_known, Y_test_rogue, axis=0)

    # Clear some memory
    X_test_known = X_test_rogue = Y_test_known = Y_test_rogue = None
    del X_test_known, X_test_rogue, Y_test_known, Y_test_rogue

    # Normalize the data to the same scale as the training set
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(-1,2)
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 2, 128)

    # Shuffle Data
    X_test, Y_test = shuffle(X_test, Y_test)

    print("Count known radios - testing: {}".format(list(Y_test).count(0)))
    print("Count rogue radios - testing: {}\n".format(list(Y_test).count(1)))

    # Evaluate the best model
    bestModel = keras.models.load_model(bestModelPath)
    scores = bestModel.evaluate(X_test, Y_test)
    foldLoss.append(scores[0])
    foldAccuracy.append(scores[1])
    bestModel.save(runPath+"//best_model_rogue_device_{}.h5".format(holdoutDevice))
    print(f"Fold Results (Best Model): Loss={scores[0]}, Accuracy={scores[1]}")

    # Graph Epoch History of Fold
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.title("Fold Training Performance - K={}".format(fold))
    plt.xlabel("Epoch Number")
    plt.savefig(runPath+"//fold-{}-training.png".format(fold))

    if fold+1 == numFolds:
        timeEnd = time.time()

        print('\n=====================Final Results=========================\n')
        for i in range(len(foldAccuracy)):
            print(f"Fold {i+1}: Loss={foldLoss[i]}, Accuracy={foldAccuracy[i]}")

        print(f"\nFold Averages: Loss={np.mean(foldLoss)}, Accuracy={np.mean(foldAccuracy)}")
        print(f"\nTotal Training Time (s): {timeEnd-timeStart}")
        print('\n===========================================================\n')    

        # Write the summary File
        writeResultsToSummary(foldLoss, foldAccuracy, timeEnd-timeStart)
