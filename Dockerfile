FROM python:3.8.6-buster

COPY TaxiFareModel /TaxiFareModel
COPY api /api
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY /home/sergeys/code/SSudakov99/gcp/sixth-oxygen-332221-3a88a79eceb2.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
