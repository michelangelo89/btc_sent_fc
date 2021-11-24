from CloudSentiment.cloud_data import get_data, transform_data
from CloudSentiment.cloud_params import LOCAL_CRYPTO_PATH, LOCAL_PATH, GS_SENT_PATH, BUCKET_NAME
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage


class Sentimenter(object):
    def __init__(self, dates_list, text_list, out_name, **kwargs):
        """df is a dataframe containing titles/tweets and dates"""
        self.tokenizer = None
        self.model = None
        self.dates = dates_list
        self.text = text_list
        self.out_name = out_name
        self.output = None

    def set_model(self):
        """loads and defines the finbert model as a class attribute"""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def tokenize(self):
        """Prepares and tokenizes the data"""
        inputs = self.tokenizer(self.text, padding = True, truncation = True, return_tensors='pt')
        return inputs
    
    def run(self):
        """Runs the model"""
        self.set_model()
        inputs = self.tokenize()
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        positive = predictions[:, 0].tolist()
        negative = predictions[:, 1].tolist()
        neutral = predictions[:, 2].tolist()

        table = {'date': self.dates,
        'text':self.text,
        'positive':positive,
        "negative":negative, 
        "neutral":neutral}
        
        out_df = pd.DataFrame(table, columns = ["date",
                                                "text",
                                                "positive",
                                                "negative",
                                                "neutral"])

        self.output = out_df

    def save_output(self, how = "google"):
        if how == "local":
            self.output.to_csv(os.path.join(LOCAL_PATH, self.out_name))
            print(f"Sentiments saved at {LOCAL_PATH}")
            return self.output
        if how == "api":
            return self.output
        if how == "google":
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(f"test/{self.out_name}")
            blob.upload_from_string(self.output.to_csv(),"text/csv")
            return self.output

    #def upload_to_gcp(self, filename):
    #    client = storage.Client()
    #    bucket = client.bucket(BUCKET_NAME)
    #    blob = bucket.blob(f"test/{filename}")
    #    blob.upload_from_string(self.output.to_csv(),"text/csv")


if __name__ == "__main__":
    N = 10
    df = get_data("crypto", nrows = N, how = "google")
    text, dates = transform_data(df)
    crypto_sentiment = Sentimenter(dates, text, out_name = "crypto_10_test.csv")
    crypto_sentiment.set_model()
    crypto_sentiment.run()
    #crypto_sentiment.upload_to_gcp("test1.csv")
    out_df = crypto_sentiment.save_output(how = "google")
    print(out_df.head())
