from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from functools import partial
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Early stopping callback based on val loss threshold
#
# Code by ZFTurbo
# https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.09, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


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
        f.write("Activation Function Output Layer: {}\n".format(activationOutput))
        f.write("Loss Function: {}\n".format(lossFunction))
        f.write("Optimizer: {}\n".format(optimizer))
        f.write("Learning Rate: {}\n".format(learningRate))
        f.write("Momentum: {}\n".format(optMomentum))
        f.write("Batch Size: {}\n\n".format(batchSize))

        f.write("Run Description\n")
        f.write("---------------\n")
        f.write("Number Cross-Validation Folds: {}\n".format(numFolds))
        f.write("Number Epochs: {}\n".format(numEpochs))
        f.write("Early Stopping Patience: {}\n".format(esPatience))
        f.write("Early Stopping Mointor: {}\n".format(esMonitor))
        f.write("Best Model Metric: {}\n\n".format(bestModelMetric))


def writeResultsToSummary(foldLoss, foldAccuracy, bestModelLoss, bestModelAccuracy, totalTime):
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

        f.write("Best Model Loss: {}\n".format(bestModelLoss))
        f.write("Best Model Accuracy: {}\n\n".format(bestModelAccuracy))

        f.write("Total trainining time (s): {}".format(totalTime))
        
# Globals
SNR = "40dB"

PATH = "F:\\Research\\Data\\Hardware Signals\\"
DEVICES = ["mRo_1", "mRo_2", "mRo_3", "3DR_T1", "3DR_TL1", "RFD900_111", "RFD900_112", "RFD900_113", "RFD900_114"]
DATASET_PATH =  PATH + "multiradio_{}-RFD900_{}\\".format(len(DEVICES), SNR)

trainingDate = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
runPath = PATH + "models\\multiradio\\" + trainingDate
bestModelPath = runPath + "\\best_model.h5"

# Create Necessary Directories
if not os.path.isdir(PATH + "models\\multiradio"):
    if not os.path.isdir(PATH + "models"):
        os.mkdir(PATH + "models")
    os.mkdir(PATH + "models\\multiradio")
os.mkdir(runPath)

#=============================================================================================

# Load Data
X_train = np.load("{}multiclass_training_samples_{}.npy".format(DATASET_PATH, SNR))
Y_train = np.load("{}multiclass_training_labels_{}.npy".format(DATASET_PATH, SNR))

X_test = np.load("{}multiclass_testing_samples_{}.npy".format(DATASET_PATH, SNR))
Y_test = np.load("{}multiclass_testing_labels_{}.npy".format(DATASET_PATH, SNR))

# Make sure data is the correct type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Shuffle Data
X_train, Y_train = shuffle(X_train, Y_train, random_state=10)
X_test, Y_test = shuffle(X_test, Y_test, random_state=10)

#=============================================================================================

# Setup model hyperparameters, training attributes
numHiddenLayers = 3
sizeHiddenLayer = 300
dropoutRate = 0.1
batchSize = 128
numEpochs = 300
activationHidden = "elu"
kernelInitializer = "he_normal"
activationOutput = "softmax"
lossFunction = "sparse_categorical_crossentropy"
optimizer = "SGD"
learningRate = 0.01
optMomentum = 0.9
esMonitor = "validation loss - minimum"
bestModelMetric = "validation accuracy - maximum"
esPatience = 15

# Setup K-Fold Cross Validation Splits
numFolds = 10
foldAccuracy = []
foldLoss = []

kfold = KFold(n_splits=numFolds, shuffle=True)

#=============================================================================================

writeSummaryFile()

fold = 1
timeStart = time.time()
for train, validate in kfold.split(X_train, Y_train):

    # Build the DNN model
    RegularizedDense = partial(keras.layers.Dense,
                            activation=activationHidden,
                            kernel_initializer=kernelInitializer, 
                            )

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[2,128]),
        RegularizedDense(sizeHiddenLayer),
        keras.layers.Dropout(rate=dropoutRate),
        RegularizedDense(sizeHiddenLayer),
        keras.layers.Dropout(rate=dropoutRate),
        RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dropout(rate=dropoutRate),
        # RegularizedDense(sizeHiddenLayer),
        # keras.layers.Dense(len(DEVICES), activation=activationOutput)
        keras.layers.Dense(2, activation=activationOutput)
    ])

    model.compile(loss = lossFunction,
                optimizer = keras.optimizers.SGD(lr=learningRate, momentum=optMomentum),
                metrics=["accuracy"])

    print('\n=============================================================')
    print(f'Training fold {fold}...\n')

    # Implement early stopping with checkpoints to curb overfitting
    esEpochs = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esPatience)
    esValLoss = EarlyStoppingByLossVal(monitor='val_loss', value=5e-5, verbose=1)
    mc = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Train the model
    history = model.fit(X_train[train], Y_train[train], validation_data=(X_train[validate], Y_train[validate]), epochs=numEpochs, batch_size=batchSize, callbacks=[esEpochs, esValLoss, mc])

    # Evaluate the best model
    bestModel = keras.models.load_model(bestModelPath)
    scores = bestModel.evaluate(X_test, Y_test)
    foldLoss.append(scores[0])
    foldAccuracy.append(scores[1])

    bestModel.save(runPath+"//best_model_fold{}.h5".format(fold))
    
    print(f"Fold Results (Best Model): Loss={scores[0]}, Accuracy={scores[1]}")

    # Graph Epoch History of Fold
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.title("Fold Training Performance - K={}".format(fold))
    plt.xlabel("Epoch Number")
    plt.savefig(runPath+"//fold-{}-training.png".format(fold))

    if fold == numFolds:
        timeEnd = time.time()

        print('\n=====================Final Results=========================\n')
        for i in range(len(foldAccuracy)):
            print(f"Fold {i+1}: Loss={foldLoss[i]}, Accuracy={foldAccuracy[i]}")

        print(f"\nFold Averages: Loss={np.mean(foldLoss)}, Accuracy={np.mean(foldAccuracy)}")
        print(f"\nTotal Training Time (s): {timeEnd-timeStart}")
        print('\n===========================================================\n')    

        # Write the summary File
        writeResultsToSummary(foldLoss, foldAccuracy, scores[0], scores[1], timeEnd-timeStart)

    fold += 1