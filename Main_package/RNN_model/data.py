import pandas as pd
import os
import numpy as np
from google.cloud import storage
from Main_package.RNN_model.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
import gcsfs
import joblib
import pickle 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_selector
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import layers, callbacks
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from Main_package.RNN_model.params import no_log_col_, target, log_col_


def get_data_from_gcp(optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path)
    return df



def get_features_from_raw_data():

    data_path = 'raw_data/final_data'

    df = pd.read_csv(os.path.join(data_path, 'features_2016.csv'),
                index_col=0,
                parse_dates=True)


    log = FunctionTransformer(lambda x: np.log(x))

    log_col = Pipeline([('log',log),('imputer', KNNImputer()),('scaler', StandardScaler())])

    target_col = Pipeline([('log',log),('imputer', KNNImputer())])

    no_log_col = Pipeline([('imputer', KNNImputer()),('scaler', StandardScaler())])

    preproc_pipe = make_column_transformer((log_col, log_col_),
                                               (no_log_col, no_log_col_),
                                               (target_col, target),
                                               remainder='passthrough')


    return pd.DataFrame(preproc_pipe.fit_transform(df),columns = df.columns,
                        index = df.index)


def clean_features(df):

    log = FunctionTransformer(np.log)
    #def log(x):
    #    return np.log(x)

    log_col = Pipeline([('log',log),('imputer', KNNImputer()),('scaler', StandardScaler())])

    target_col = Pipeline([('log',log),('imputer', KNNImputer())])

    no_log_col = Pipeline([('imputer', KNNImputer()),('scaler', StandardScaler())])

    preproc_pipe = make_column_transformer(
        (log_col, log_col_),
        (no_log_col, no_log_col_),
        (target_col, target),
        remainder= no_log_col)
    
    #preproc_pipe = Pipeline(["columns", columns])
    preproc_pipe.fit(df)
    joblib.dump(preproc_pipe, "../pipe.joblib")

    return pd.DataFrame(preproc_pipe.transform(df),
                       columns = df.columns,
                       index = df.index)

def clean_test_features(df):
    preproc_pipe = joblib.load("../pipe.joblib")
    return pd.DataFrame(preproc_pipe.transform(df),
                        columns = df.columns,
                        index = df.index)




def get_target_from_raw_data():
    data_path = '../../raw_data/final_data'

    return pd.read_csv(os.path.join(data_path, 'target_GV.csv'),
                           index_col=0 ,parse_dates=True)

if __name__ == '__main__':
    df = get_features_from_raw_data()
