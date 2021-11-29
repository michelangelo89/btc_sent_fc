from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from CloudSentiment.cloud_trainer import Sentimenter
from CloudSentiment.cloud_data import get_data, transform_data
from CloudSentiment.cloud_tweet_scraper import TweetScraper
import datetime as dt
import pytz
import joblib
import os
import ast
from time import sleep

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
def scrape_twitter(n=1, topic = "inflation"):
    n = int(n)
    LIST_DATES = []
    date = dt.datetime(2021,11,29,0,0)
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
            sentiment = Sentimenter(date_list, text_list, out_name = f"tweet_inflation_{LIST_DATES[i]}.csv")
            sentiment.set_model()
            sentiment.run()
            out_df = sentiment.save_output("google")
        sleep(5)
        print(f"{LIST_DATES[i]} processed")



