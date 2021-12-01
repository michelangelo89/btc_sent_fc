import os
from math import sqrt
import numpy as np
import joblib
import pandas as pd
from google.cloud import storage
from Main_package.RNN_model.data import clean_features


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline



#dirname = os.path.dirname(__file__)
#PATH_TO_MODEL = os.path.join(dirname, "..", "model_RNN_1.joblib")

def predict_one_day(
        PATH_TO_MODEL="model.joblib",
        shape=(1, 90, 67),
        train_path="raw_data/test_2021_11_22.csv"):
    """
    train_path shape must be 90 rows 67 columns
    """
    model = joblib.load(PATH_TO_MODEL)
    X_pred = np.zeros(shape)
    X_pred[0] = np.array(
        pd.read_csv("raw_data/test_2021_11_22.csv",
                    index_col=0,
                    parse_dates=True))
    return model.predict_on_batch(X_pred[0])
