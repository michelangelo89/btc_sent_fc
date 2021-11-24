from cloud_data import get_data, transform_data
from cloud_params import LOCAL_CRYPTO_PATH, LOCAL_PATH
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    
    def save_output(self, how = "local"):
        if how == "local":
            self.output.to_csv(os.path.join(LOCAL_PATH, self.out_name))
            print(f"Sentiments saved at {LOCAL_PATH}")
            return self.output
#   def upload_to_gcp(self, filename):
#       client = storage.Client()
#       bucket = client.bucket(BUCKET_NAME)
#       blob = bucket.blob("models/recapmodel/TEMPFILENAME.joblib")
#       filename
#       blob.upload_from_filename(filename)

if __name__ == "__main__":
    N = 10
    df = get_data("crypto", nrows = N)
    text, dates = transform_data(df)
    crypto_sentiment = Sentimenter(dates, text, out_name = "crypto_10_test")
    crypto_sentiment.set_model()
    crypto_sentiment.run()
    out_df = crypto_sentiment.save_output()
    print(out_df.head())




