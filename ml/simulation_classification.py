#import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Globals
noiseAmp = 0.01
PATH = "/media/ericmiers/Grad School Data/Research/Data/Simulated Signals/"
DEVICES = ["dev_0", "dev_1", "dev_2", "dev_3"]

#=============================================================================================

# Load and shuffle Training Data
X_train = np.load("{}{}/samples_{}.npy".format(PATH, "train", noiseAmp))
Y_train = np.load("{}{}/labels_{}.npy".format(PATH, "train", noiseAmp))

# X_train, Y_train = shuffle(X_train, Y_train)

# Load Testing Data
X_test = np.load("{}{}/samples_{}.npy".format(PATH, "test", noiseAmp))
Y_test = np.load("{}{}/labels_{}.npy".format(PATH, "test", noiseAmp))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#=============================================================================================

# Setup K-Fold Cross Validation Splits
numFolds = 10
foldAccuracy = []
foldLoss = []

inputs = np.concatenate((X_train, X_test))
targets = np.concatenate((Y_train, Y_test))

kfold = KFold(n_splits=numFolds, shuffle=True)

#=============================================================================================

fold = 1
for train, test in kfold.split(inputs, targets):
    # Build the DNN model
    RegularizedDense = partial(keras.layers.Dense,
                            activation="relu"
                            # kernel_initializer="he_normal", 
                            # kernel_regularizer=keras.regularizers.l2(0.01)
                            )

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[2,128]),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        RegularizedDense(300),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        RegularizedDense(100),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        # RegularizedDense(100),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        # RegularizedDense(100),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9),
                metrics=["accuracy"])

    print('\n=============================================================')
    print(f'Training fold {fold}...\n')

    # Train the model
    history = model.fit(inputs[train], targets[train], epochs=25)

    # Test the model
    scores = model.evaluate(inputs[test], targets[test])
    foldLoss.append(scores[0])
    foldAccuracy.append(scores[1] * 100)
    
    print(f"Fold Results: Loss={scores[0]}, Accuracy={scores[1]}")

    fold += 1

print('=====================Final Results=========================\n')
for i in range(len(foldAccuracy)):
    print(f"Fold {i+1}: Loss={foldLoss[i]}, Accuracy={foldAccuracy[i]}")

print(f"\nFold Averages: Loss={np.mean(foldLoss)}, Accuracy={np.mean(foldAccuracy)}")
print('\n===========================================================\n')    

model.save(f"{PATH}models/12_1_2020_10folds.h5")

# # Graph results
# pd.DataFrame(history.history).plot(figsize=(8,5))
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()