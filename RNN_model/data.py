import pandas as pd
import os

from google.cloud import storage
from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH



def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_data(df, test=False):

    pass


def get_features_from_raw_data():
    data_path = '../raw_data/final_data'

    return pd.read_csv(os.path.join(data_path, 'features_2016.csv'),
                           index_col=0 ,parse_dates=True)


def get_target_from_raw_data():
    data_path = '../raw_data/final_data'

    return pd.read_csv(os.path.join(data_path, 'target_GV.csv'),
                           index_col=0 ,parse_dates=True)

if __name__ == '__main__':
    df = get_features_from_raw_data()
