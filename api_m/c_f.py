from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os
import gcsfs
from Main_package.RNN_model.data import clean_features
from Main_package.RNN_model.predict import predict_one_day

dirname = os.path.dirname(__file__)
PATH_TO_MODEL = os.path.join(dirname, "..", "RNN_2_MAPE_best.joblib")

fs = gcsfs.GCSFileSystem()
with fs.open('wagon-data-750-btc-sent-fc/model/finbert_token.joblib') as f:
    model = joblib.load(f)


app_m = FastAPI()


@app_m.get("/")
def index():
    return{"okkk": True}

@app_m.get("/predict")
def predict():
    y_pred = predict_one_day(PATH_TO_MODEL="model.joblib",
                                   shape=(1, 90, 67),
                                   train_path="raw_data/test_2021_11_22.csv")
    return {"prediction": f"{np.exp(y_pred[0][0])}"}
    #return "It works"
