import requests
from cloud_params import LOCAL_CRYPTO_PATH, LOCAL_ECON_PATH
import pandas as pd
import numpy as np

def get_data(how, nrows = 100):
    """How is a string that can equal "econ", "crypto", or "twitter"
    Returns the approrpriate dataset as a pandas dataframe"""
    if how == "econ":
        df = pd.read_csv(LOCAL_ECON_PATH, nrows = nrows)
        return df[["date","title"]]
    if how == "crypto":
        df =  pd.read_csv(LOCAL_CRYPTO_PATH, nrows = nrows)
        return df[["date","title"]]
    
    print("Input one of the following strings: 'crypto', 'econ', or 'twitter'")
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