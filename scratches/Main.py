#Adib Thaqif
#Income Predictor
from __future__ import print_function
import math
from IncomePredictor import my_input_fn
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from PreprocessFeatures import preprocess_features, preprocess_targets, intializeDataSet
from trainModel import train_model


income_classification_dataframe = pd.read_csv("income_evaluation.csv", sep=",")  # type: Any

#covers things like resizing, randomizing, and formatting features
income_classification_dataframe = intializeDataSet(income_classification_dataframe)

#stores the number of rows in my dataframe
dfSize = len(income_classification_dataframe.index)

#training data will use the first 70% of the dataframe
training_examples = preprocess_features(income_classification_dataframe.head(math.floor(dfSize * 0.7)))
training_targets = preprocess_targets(income_classification_dataframe.head(math.floor(dfSize * 0.7)))

#test data will use the remaining 30% of the dataframe
test_examples = preprocess_features(income_classification_dataframe.tail(math.floor(dfSize * 0.3)))
test_targets = preprocess_targets(income_classification_dataframe.tail(math.floor(dfSize * 0.3)))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())
train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=4
)