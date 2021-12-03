from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from Main_package.CloudSentiment.cloud_trainer import Sentimenter
from Main_package.CloudSentiment.cloud_data import get_data, transform_data
from Main_package.CloudSentiment.cloud_tweet_scraper import TweetScraper, list_blobs
import datetime as dt
import pytz
import joblib
import os
import ast
from time import sleep
import numpy as np
import gcsfs


dirname = os.path.dirname(__file__)
PATH_TO_MODEL = os.path.join(dirname, "..", "model_RNN_1.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/bert")
def sentiment(date_list,
            text_list,
            out_name = "test_api"):
    #if N:
    #    N = int(N)
    #df = get_data("crypto", nrows = N, how = "google")
    #text_list, date_list = transform_data(df)
    text_list = ast.literal_eval(text_list)
    date_list = ast.literal_eval(date_list)
    sentiment = Sentimenter(date_list, text_list, out_name = out_name+".csv")
    sentiment.set_model()
    sentiment.run()
    out_df = sentiment.save_output("google")
    output = out_df.to_json()

    return out_df

@app.get("/blobs")
def return_blob(topic = "inflation"):
    return list_blobs(topic)[0]

@app.get("/tweet")
def scrape_twitter(n=1, start_date = None, topic = "inflation"):
    if start_date:
        date = dt.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.000Z")
    else:
        start_date = list_blobs(topic)[0]
    n = int(n)
    LIST_DATES = []
    date = dt.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.000Z")
    for i in range(n):
        LIST_DATES.append(date.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        date = date - dt.timedelta(days = 1)
    for i in range(len(LIST_DATES)-1):
        scraper = TweetScraper(LIST_DATES[i],
                            LIST_DATES[i+1],
                                topic)
        scraper.set_keys()
        scraper.get_tweets_dict()
        scraper.clean_df()
        tweet_df = scraper.save_df()
        for k in range(0, tweet_df.shape[0],20):
            df = tweet_df.iloc[k:k+20].copy()
            text_list, date_list = transform_data(df[["tweet_date","title"]])
            sentiment = Sentimenter(date_list, text_list, out_name = f"tweet_inflation_{LIST_DATES[i]}_{k}.csv")
            sentiment.set_model()
            sentiment.run()
            out_df = sentiment.save_output("google")
        sleep(5)
        #print(f"{LIST_DATES[i]} processed")


@app.get("/predict")
def predict(model_name = "model_RNN_8.joblib", file_name = "pred_1.csv", shape_row = 90, shape_feat = 61):
    """model_name is a string - "model_name.joblib"
    date is also a string - in the format "yyyy_mm_dd"
    shape is a 3 part tuple with the input dimensions of the model"""
    shape = (1, int(shape_row), int(shape_feat))
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'wagon-data-750-btc-sent-fc/model/{model_name}') as f:
        model = joblib.load(f)

    X_pred = np.zeros(shape)
    X_pred[0] = pd.read_csv(
        f"gcs://wagon-data-750-btc-sent-fc/input_data/{file_name}",
        index_col = 0,
        parse_dates = True)
    y_pred = model.predict_on_batch(X_pred)
    return {"prediction": f"{np.exp(y_pred[0][0])}"}
    #return "It works"

