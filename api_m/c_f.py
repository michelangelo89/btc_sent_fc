from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os
import gcsfs
from Main_package.RNN_model.data import clean_features

dirname = os.path.dirname(__file__)
PATH_TO_MODEL = os.path.join(dirname, "..", "model_RNN_8.joblib")

fs = gcsfs.GCSFileSystem()
with fs.open('wagon-data-750-btc-sent-fc/model/finbert_token.joblib') as f:
    model = joblib.load(f)


app_m = FastAPI()


@app_m.get("/")
def index():
    return{"okkk": True}

@app_m.get("/predict")
def predict():
    model = joblib.load(PATH_TO_MODEL)
    X_pred = np.zeros((1, 89, 61))
    X_pred[0] = np.array(
        clean_features(pd.read_csv("raw_data/Michelangelo_test.csv",
                                  index_col=0,
                                  parse_dates=True)))
    y_pred = model.predict_on_batch(X_pred)
    return {"prediction": f"{np.exp(y_pred[0][0])}"}
    #return "It works"
