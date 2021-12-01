import joblib
from termcolor import colored
import mlflow
import pandas as pd
import numpy as np

from Main_package.RNN_model.data import get_features_from_raw_data, get_target_from_raw_data
from Main_package.RNN_model.utils import get_X_y
from Main_package.RNN_model.gcp import storage_upload
from Main_package.RNN_model.RNN_model import initial_model
from Main_package.RNN_model.params import MLFLOW_URI, EXPERIMENT_NAME, BUCKET_NAME, MODEL_VERSION, MODEL_VERSION
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_selector
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import layers, callbacks
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from Main_package.RNN_model.params import no_log_col_, target, log_col_



class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name


    def run_model(self):

        es = callbacks.EarlyStopping(patience=20)
        self.model = initial_model()
        self.model.fit(self.X , self.y,
                            validation_split = 0.2,
                            epochs=3,
                            callbacks=[es],verbose=1)
        return self.model


    #def run(self):
    #    self.set_pipeline()
    #self.mlflow_log_param("model", "Linear")

    #   self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        res = self.model.evaluate(X_test, y_test)
        #y_pred = self.pipeline.predict(X_test)
        #rmse = compute_rmse(y_pred, y_test)
        # self.mlflow_log_metric("MAPE", res)
        return res #print(f'MAPE on the test set : {res[2]:.0f} %')


    def save_model_locally(self,name='model'):
        """Save the model into a .joblib format"""
        joblib.dump(self.model, f'{name}.joblib')
        print(colored(f"{name}.joblib saved locally", "green"))



if __name__ == "__main__":
    # Get and clean data
    #N = 100
    df = get_features_from_raw_data()  #nrows=N)
    #df = clean_data(df)
    len_ = int(0.8 * df.shape[0])
    df_train = df[:len_]
    df_test = df[len_:]
    X_train, y_train = get_X_y(df_train, 2000, 121, target_name='volume_gross')
    X_test, y_test = get_X_y(df_test, 2000, 121, target_name='volume_gross')
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    #trainer.set_experiment_name('RNN_BTC')
    trainer.run_model()
    res = trainer.evaluate(X_test, y_test)
    print(f'MAPE on the test set : {res[2]:.0f} %')
    #print(f"rmse: {res}")
    trainer.save_model_locally()

    #storage_upload()
