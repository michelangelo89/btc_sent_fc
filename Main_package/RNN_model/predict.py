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



dirname = os.path.dirname(__file__)
PATH_TO_MODEL = os.path.join(dirname, "..", "..", "RNN_2_MAPE_best.joblib")

dirname = os.path.dirname(__file__)
PATH_TO_DATA = os.path.join(dirname, "..", "..", "raw_data/test_log_vol.csv")


def predict_one_day(
        shape=(1, 90, 67)):
    """
    train_path shape must be 90 rows 67 columns
    """
    #preproc = joblib.load('preproc_pipe.joblib')
    model = joblib.load(PATH_TO_MODEL)
    X_pred = np.zeros(shape)
    df = pd.read_csv(PATH_TO_DATA, index_col=0, parse_dates=True)
    df = df.interpolate(method='linear', axis=0)
    #df['volume_gross'] = np.log(df['volume_gross'])
    X_pred[0] = np.array(df)
    return np.exp(model.predict_on_batch(X_pred[0][0]))

if __name__ == "__main__":
    """
    train_path shape must be 90 rows 67 columns
    """
    #preproc = joblib.load('preproc_pipe.joblib')
    model = joblib.load(PATH_TO_MODEL)
    X_pred = np.zeros((1, 90, 67))
    df = pd.read_csv(PATH_TO_DATA, index_col=0, parse_dates=True)
    df = df.interpolate(method='linear', axis=0)
    #df['volume_gross'] = np.log(df['volume_gross'])
    X_pred[0] = np.array(df)
    print(np.exp(model.predict_on_batch(X_pred[0][0])))
