from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from CloudSentiment.cloud_trainer import Sentimenter
from CloudSentiment.cloud_data import get_data, transform_data
from CloudSentiment.cloud_tweet_scrapert import TweetScraper
from datetime import datetime
import pytz
import joblib
import os
import ast

dirname = os.path.dirname(__file__)
PATH_TO_MODEL = os.path.join(dirname, "..", "model.joblib")

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
def predict(date_list,
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

@app.get("/tweet")
def scrape_twitter():
    scraper = TweetScraper()
    scraper = TweetScraper('2021-11-25T00:00:00.000Z',
                                "2021-11-26T00:00:00.000Z",
                                "economy")
    scraper.set_keys()
    scraper.get_tweets_dict()
    scraper.clean_df()