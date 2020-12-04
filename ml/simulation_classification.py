from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Globals
PATH = "/media/ericmiers/Grad School Data/Research/Data/Simulated Signals/"
DEVICES = ["dev_0", "dev_1", "dev_2", "dev_3"]

noiseAmp = 0.01
bestModelPath = PATH + 'models/best_model.h5'
numEpochs = 130
#=============================================================================================

# Load Data
X = np.load("{}samples_{}.npy".format(PATH, noiseAmp))
Y = np.load("{}labels_{}.npy".format(PATH, noiseAmp))

# Make sure data is the correct type
X = X.astype('float32')

# Shuffle Data
X, Y = shuffle(X, Y)

#=============================================================================================

# Setup K-Fold Cross Validation Splits
numFolds = 10
foldAccuracy = []
foldLoss = []

kfold = KFold(n_splits=numFolds, shuffle=True)

#=============================================================================================

fold = 1
for train, validate in kfold.split(X, Y):
    # Build the DNN model
    RegularizedDense = partial(keras.layers.Dense,
                            activation="elu",
                            kernel_initializer="he_normal", 
                            # kernel_regularizer=keras.regularizers.l2(0.01)
                            )

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[2,128]),
        #keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        RegularizedDense(300),
        #keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        RegularizedDense(300),
        #keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        RegularizedDense(300),
        #keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        #RegularizedDense(100),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9),
                #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                metrics=["accuracy"])


    print('\n=============================================================')
    print(f'Training fold {fold}...\n')

    # Implement early stopping with checkpoints to curb overfitting
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    mc = keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # Train the model
    history = model.fit(X[train], Y[train], validation_data=(X[validate], Y[validate]), epochs=numEpochs, callbacks=[es, mc])

    # Validate the best model
    bestModel = keras.models.load_model(bestModelPath)
    scores = bestModel.evaluate(X[validate], Y[validate])
    foldLoss.append(scores[0])
    foldAccuracy.append(scores[1] * 100)
    
    print(f"Fold Results (Best Model): Loss={scores[0]}, Accuracy={scores[1]}")

    if fold == 10:

        print('\n=====================Final Results=========================\n')
        for i in range(len(foldAccuracy)):
            print(f"Fold {i+1}: Loss={foldLoss[i]}, Accuracy={foldAccuracy[i]}")

        print(f"\nFold Averages: Loss={np.mean(foldLoss)}, Accuracy={np.mean(foldAccuracy)}")
        print(f"\nTest on novel signal: Loss={testScores[0]}, Accuracy={testScores[1]}")
        print('\n===========================================================\n')    

        bestModel.save(f"{PATH}models/12_4_2020_10folds.h5")

        # Graph Epoch History of Final Fold
        pd.DataFrame(history.history).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(0,1)
        plt.show()

    fold += 1