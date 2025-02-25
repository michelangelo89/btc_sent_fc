FROM python:3.8.6-buster

COPY Main_package/CloudSentiment Main_package/CloudSentiment
COPY api /api
COPY requirements.txt /requirements.txt
COPY finbert_model.joblib /finbert_model.joblib
COPY finbert_token.joblib /finbert_token.joblib
COPY raw_data/test_2021_11_22.csv /raw_data/test_2021_11_22.csv
COPY keys.json /keys.json
COPY btc-sent-fc-dd81034dfecd.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.cloud_fast:app --host 0.0.0.0 --port $PORT
