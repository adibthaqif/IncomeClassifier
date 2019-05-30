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

pd.__version__

#needs more research
def intializeDataSet(income_classification_dataframe):


    # randomize data
    income_classification_dataframe = income_classification_dataframe.reindex(
        np.random.permutation(income_classification_dataframe.index)
    )



#needs more research
def preprocess_features(income_classification_dataframe):
    """Prepares input features from income_classification data set.
       Needs to use one hot encoding to represent features like occupation, marital-status,
       , native country, etc..
    Args:
      income_classification_dataframe: A Pandas DataFrame expected to contain data
        from the airfare_report data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """

    selected_features = income_classification_dataframe[
        ["age",
         "education",
         "relationship",
         "race",
         "sex",
         "capital-gain",
         "capital-loss",
         "hours-per-week",
         "native-country",]]

    #use one hot encoding here for certain categorical features

    processed_features = selected_features.copy()
    # Create a synthetic feature to add to the existing features
    # needs more research

    return processed_features

#needs more research
def preprocess_targets(income_classification_dataframe):
    """Prepares target features (i.e., labels) from income_classification set.
    Args:
      income_classification_dataframe: A Pandas DataFrame expected to contain data
        from income_classification data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()

    output_targets["income"] = (
        income_classification_dataframe["income"])
    return output_targets