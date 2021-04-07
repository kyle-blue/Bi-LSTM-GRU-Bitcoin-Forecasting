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

from app.parameters import Architectures
from .RSquaredMetric import RSquaredMetric
from sklearn.preprocessing import MinMaxScaler
from app.test_model import test_model


# DATASET = Dataset.WEATHER.value
# MAX_DATASET_SIZE = 100000 # Dataset has over 1 mil values, so we limit to last x values
# SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value # The current symbol to train the model to base predictions on

# TODO: I don't think we need future period, max dataset size, seq_len, symbol_to_predict, dataset. This is the job of DataPreprocesser 
class Model():
    def __init__(self, *,  max_epochs = 100, batch_size = 1024, hidden_layers = 2,
        neurons_per_layer = 64, model = Architectures.LSTM.value, dropout = 0.1,
        is_bidirectional = False, initial_learn_rate = 0.001):
        

    