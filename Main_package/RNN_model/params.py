### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[UK] [LONDON] [DALESSANDRO] LinearRegression + 0.1"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-750-dalessandro1989'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


no_log_col_ = [
    'tweets_sent', 'reddit_crypto_sent', 'reddit_econ_sent', 'News Sentiment',
    'mempool-size']

target = ['volume_gross']

log_col_ = [
 'n-transactions-per-block',
 'difficulty',
 'utxo-count',
 'mvrv',
 'nvt',
 'avg-block-size',
 'n-transactions-excluding-popular',
 'n-unique-addresses',
 'median-confirmation-time',
 'miners-revenue',
 'mempool-growth',
 'blocks-size',
 'hash-rate',
 'n-transactions-total',
 'avg-confirmation-time',
 'nvts',
 'transaction-fees-usd',
 'active_account',
 'Russell 2000 Index (RUT) - Index Value',
 'CBOE Volatility S&P 500 Index (^VIX) - Index Value',
 'S&P 500 (^SPX) - Index Value',
 'NASDAQ Composite Index (^COMP) - Index Value',
 'Dow Jones Industrial Average (^DJI) - Index Value',
 'S&P U.S. Treasury Bill 0-3 Month Index',
 'S&P U.S. Treasury Bond 10+ Year Index',
 'S&P U.S. TIPS 5+ Year Index (USD)',
 'S&P U.S. Treasury Bond 1-3 Year Index',
 'S&P Canada Treasury Bill Index',
 'S&P U.S. Treasury Bond Current 30-Year Index',
 'S&P U.S. Treasury Bond Current 5-Year Index',
 'S&P Short Term Taxable Municipal Bond Index',
 'S&P Taxable Municipal Bond Index',
 'S&P U.S. Treasury Bond Current 2-Year Index',
 'S&P U.S. Treasury Bond 7-10 Year Index',
 'S&P U.S. TIPS 0-1 Year Index (USD)',
 'S&P U.S. Treasury Bond 20+ Year Index',
 'S&P U.S. Government & Corporate AAA-AA 1+ Year Bond Index',
 'S&P Municipal Yield Index',
 'S&P U.S. Treasury Bond Current 7-Year Index',
 'S&P U.S. Treasury Bond Current 10-Year Index',
 'S&P U.S. Treasury Bill 9-12 Month Index',
 'S&P U.S. Treasury Bond Current 3-Year Index',
 'S&P U.S. TIPS 30 Year Index (USD)',
 'S&P U.S. Ultra Short Treasury Bill & Bond Index (USD)',
 'S&P U.S. TIPS 15+ Year Index (USD)',
 'S&P U.S. Treasury Bond 5-7 Year Index',
 'S&P U.S. TIPS 10+ Year Index (USD)',
 'S&P U.S. Treasury Bill 3-6 Month Index',
 'S&P U.S. Treasury Bond 7-10 Year Index (TTM JPY)',
 'S&P U.S. TIPS 7-10 Year Index (USD)',
 'S&P U.S. Treasury Bond 3-5 Year Index',
 'S&P U.S. TIPS 3-5 Year Index (USD)',
 'S&P U.S. TIPS 1-3 Year Index (USD)',
 'S&P U.S. TIPS 5-7 Year Index (USD)',
 'S&P U.S. Treasury Bill 6-9 Month Index']
