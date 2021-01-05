from tensorflow import keras
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from datetime import datetime
from generate_evaluation_summary import plot_confusion_matrix, write_summary_file
import numpy as np
import matplotlib
import os
import time


# Setup PATHs
PATH = "F:\\Research\\Data\\Hardware Signals\\"
MODEL = "01-04-2021_12-56-56"
MODEL_PATH = "F:\\Research\\Data\\Hardware Signals\\models\\" + MODEL 
MODEL_FILE = MODEL_PATH + "\\best_model.h5"

# Set SNR of interest
SNR = "40dB"

# Confusion Matrix Parameters
title = "Confusion Matrix, {}".format(SNR)
filenameCM = MODEL_PATH +"\\confusion-matrix_{}.png".format(SNR)
filenameSummary = MODEL_PATH +"\\model-evaluation-summary_{}.txt".format(SNR)
labels = [0,1,2,3,4,5,6,7]
normalize = False # True = use prediction percentages, False = use prediction counts

# Load the model
model = keras.models.load_model(MODEL_FILE)

X_test = np.load("{}testing_samples_{}.npy".format(PATH, SNR))
Y_test = np.load("{}testing_labels_{}.npy".format(PATH, SNR))

# Make sure data is the correct type
X_test = X_test.astype('float32')

# Shuffle Data
X_test, Y_test = shuffle(X_test, Y_test)

# Make predictions and evaluate with confusion matrix
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

cm = confusion_matrix(Y_test, predictions)

accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy

# F1 Metrics calculated differently depending on model type
if len(labels) > 2:
    precision = precision_score(Y_test, predictions, average="weighted")
    recall = recall_score(Y_test, predictions, average="weighted")
    F1 = f1_score(Y_test, predictions, average="weighted")
else:
    precision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    F1 = f1_score(Y_test, predictions)

# Plot the matrix and save to file
title +="\nF1 = {}".format(F1)
plot_confusion_matrix(cm, normalize=normalize, target_names=labels, title=title, filename=filenameCM)

# Write the summary File
write_summary_file(filenameSummary, SNR, accuracy, misclass, precision, recall, F1)
