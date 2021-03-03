from tensorflow import keras
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from datetime import datetime
from generate_evaluation_summary import plot_confusion_matrix, write_summary_file
import numpy as np
import matplotlib
import os
import time

np.set_printoptions(threshold=np.inf)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
configuration = "GPU"

# Setup PATHs
PATH = "F:\\Research\\Data\\Hardware Signals\\"
DATASET = "multiradio_RFD900-devices_40dB\\"
DATASET_PATH = PATH + DATASET
MODEL_TYPE = "multiradio"
MODEL = "02-23-2021_22-25-00 (RFD900)"
MODEL_PATH = "F:\\Research\\Data\\Hardware Signals\\models\\" + MODEL_TYPE + "\\" + MODEL 
MODEL_FILE = MODEL_PATH + "\\best_model.h5"

# Set SNR of interest
SNR = "2dB"

# Load the model
model = keras.models.load_model(MODEL_FILE)

# Establish additional parameters depending on model type
if MODEL_TYPE == "binary":
    labels = [0,1]
    KNOWN_DEVICE = "mRo_1"
    ROGUE_DEVICE = "mRo_2"

    # Create Output directories for specific model evaluations
    RESULTS_PATH = MODEL_PATH + "\\{}_{}".format(KNOWN_DEVICE, ROGUE_DEVICE)
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    X_test = np.load("{}binary_prediction_samples_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, ROGUE_DEVICE, SNR))
    Y_test = np.load("{}binary_prediction_labels_{}_{}_{}.npy".format(DATASET_PATH, KNOWN_DEVICE, ROGUE_DEVICE, SNR))

    filenameCM = RESULTS_PATH +"\\confusion-matrix_{}.png".format(SNR)
    filenameSummary = RESULTS_PATH +"\\model-evaluation-summary_{}.txt".format(SNR)

else:

    # Create Output directories for specific model evaluations
    RESULTS_PATH = MODEL_PATH + "\\{}".format("evaluation")
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    labels = [0,1,2,3]
    X_test = np.load("{}multiclass_prediction_samples_{}.npy".format(DATASET_PATH, SNR))
    Y_test = np.load("{}multiclass_prediction_labels_{}.npy".format(DATASET_PATH, SNR))

    filenameCM = RESULTS_PATH +"\\confusion-matrix_{}.png".format(SNR)
    filenameSummary = RESULTS_PATH +"\\model-evaluation-summary_{}.txt".format(SNR)

# Make sure data is the correct type
X_test = X_test.astype('float32')

# Shuffle Data
X_test, Y_test = shuffle(X_test, Y_test, random_state=10)

# Make predictions and evaluate with confusion matrix
num_predictions = Y_test.shape
start = time.time()
predictions = model.predict(X_test)
end = time.time()
total_time  = end - start

# If the model is binary, transform predictions from probabilities to binary labels
if MODEL_TYPE == "binary":
    predictions = [1 * (x[0]>=0.5) for x in predictions]
else:
    predictions = np.argmax(predictions, axis=1)

cm = confusion_matrix(Y_test, predictions)

accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy

# F1 Metrics calculated differently depending on model type
if MODEL_TYPE == "binary":
    precision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    F1 = f1_score(Y_test, predictions)
else:
    precision = precision_score(Y_test, predictions, average="weighted")
    recall = recall_score(Y_test, predictions, average="weighted")
    F1 = f1_score(Y_test, predictions, average="weighted")

# Plot the matrix and save to file
title = "Confusion Matrix, {}\nF1 = {}".format(SNR, F1)
normalize = False # True = use prediction percentages, False = use prediction counts
plot_confusion_matrix(cm, normalize=normalize, target_names=labels, title=title, filename=filenameCM)

# Write the summary File
write_summary_file(filenameSummary, SNR, accuracy, misclass, precision, recall, F1, num_predictions, total_time, configuration)
