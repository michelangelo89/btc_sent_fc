import requests
from Main_package.CloudSentiment.cloud_params import LOCAL_CRYPTO_PATH, LOCAL_ECON_PATH, GS_DATA_ECON_PATH, GS_DATA_CRYPTO_PATH
import pandas as pd
import numpy as np
import os

def get_data(what, nrows = 100, how = "local"):
    """what is a string that can equal "econ", "crypto", or "twitter"
    how defines whether the data is taken locally or from 
    Returns the approrpriate dataset as a pandas dataframe"""
    if how == "local":
        if what == "econ":
            df = pd.read_csv(LOCAL_ECON_PATH, nrows = nrows)
            return df[["date","title"]]
        if what == "crypto":
            df =  pd.read_csv(LOCAL_CRYPTO_PATH, nrows = nrows)
            return df[["date","title"]]

    if how == "google":
        if what == "econ":
            df = pd.read_csv(GS_DATA_ECON_PATH, nrows = nrows)
            return df[["date","title"]]
        if what == "crypto":
            df =  pd.read_csv(GS_DATA_CRYPTO_PATH, nrows = nrows)
            return df[["date","title"]]
        print("Input one of the following strings for 'what': 'crypto', 'econ', or 'twitter'")
        pass

    print("Input one of the following strings for 'how': 'local', 'google'")
    pass



def transform_data(df):
    """Currently working with 'test' dataset of 100"""
    titles_df = df.copy()
    #titles_df = titles_df.sample(100, random_state=42
    titles_array = np.array(titles_df)
    np.random.shuffle(titles_array)
    titles_list = list(titles_array[:,1])
    dates_list = list(titles_array[:,0])
    return titles_list, dates_list