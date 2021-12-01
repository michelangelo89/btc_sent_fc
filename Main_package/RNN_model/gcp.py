import os

from google.cloud import storage
from termcolor import colored
from Main_package.RNN_model.params import BUCKET_NAME


def storage_upload(rm=False, local_model_name='model.RNN_01'):
    client = storage.Client().bucket(BUCKET_NAME)

    storage_location = f"models//{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print(
        colored(
            f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}",
            "green"))
    if rm:
        os.remove('model.joblib')
