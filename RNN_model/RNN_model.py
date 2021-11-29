import warnings
import os

import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers, callbacks


normalizer = Normalization()
normalizer.adapt(X_train)



def initial_model():

    model = Sequential()
    model.add(normalizer)

    model.add(LSTM(units=50, activation='tanh',
                   return_sequences=True))  # dropout = 0.2
    #model.add(Dropout(0.2))
    model.add(LSTM(units=100, activation='tanh', return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(loss='mae',
                  optimizer='adam',
                  metrics=[MeanAbsoluteError(),
                           MeanAbsolutePercentageError()])

    return model