FROM python:3.8.6-buster

COPY CloudSentiment /CloudSentiment
COPY api /api
COPY requirements.txt /requirements.txt
COPY btc-sent-fc-6f09fb12ae2d.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.cloud_fast:app --host 0.0.0.0 --port $PORT
