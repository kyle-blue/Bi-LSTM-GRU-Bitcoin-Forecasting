from datetime import datetime
from typing import Deque
from numpy.core.numeric import NaN
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import pandas as pd
from collections import deque
import os
import random
import ta
from .parameters import *
from .RSquaredMetric import RSquaredMetric
from sklearn.preprocessing import MinMaxScaler
from app.test_model import test_model

### SEQ INFO
DATASET = Dataset.WEATHER.value
MAX_DATASET_SIZE = 100000 # Dataset has over 1 mil values, so we limit to last x values
SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value # The current symbol to train the model to base predictions on
FUTURE_PERIOD = 30 # The look forward period for the future column, used to train the neural network to predict future price
SEQUENCE_LEN = 200 # The look back period aka the sequence length. e.g if this is 100, the last 100 prices will be used to predict future price

EPOCHS = 100 # Epochs per training fold (we are doing 10 fold cross validation)
BATCH_SIZE = 1024

## MODEL INFO
HIDDEN_LAYERS = 4
NEURONS_PER_LAYER = 64
MODEL = Architectures.GRU.value
DROPOUT = 0.0


class DataPreprocesser():
    def __init__(self, dataset_path: str, forecast_col_name:str, *,
        max_dataset_size = 100000, forecast_period = 1, sequence_length = 100):
        pass