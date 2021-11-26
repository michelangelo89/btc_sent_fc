import os

dirpath = os.getcwd()
LOCAL_PATH = os.path.join(dirpath,"..","raw_data")
LOCAL_CRYPTO_PATH = os.path.join(dirpath,"..","raw_data","crypto_reddit.csv")
LOCAL_ECON_PATH = os.path.join(dirpath,"..","raw_data", "crypto_econ_prelim.csv")
BUCKET_NAME = "wagon-data-750-btc-sent-fc"
BUCKET_SENT_FOLDER = "sent_data"
BUCKET_DATA_FOLDER = "raw_data"
GS_DATA_CRYPTO_PATH = f"gcs://{BUCKET_NAME}/{BUCKET_DATA_FOLDER}/crypto_reddit.csv"
GS_DATA_ECON_PATH = f"gcs://{BUCKET_NAME}/{BUCKET_DATA_FOLDER}/crypto_econ_prelim.csv"
GS_SENT_PATH = f"gcs://{BUCKET_NAME}/{BUCKET_SENT_FOLDER}/"