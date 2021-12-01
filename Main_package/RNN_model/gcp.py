import os

from google.cloud import storage
from termcolor import colored
from Main_package.RNN_model.params import BUCKET_NAME


def storage_upload(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.RNN_01'
    storage_location = f"models//{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print(
        colored(
            f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
            "green"))
    if rm:
        os.remove('model.joblib')
