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

from app.parameters import Architecture
from .RSquaredMetric import RSquaredMetric
from sklearn.preprocessing import MinMaxScaler
from app.test_model import test_model


# DATASET = Dataset.WEATHER.value
# MAX_DATASET_SIZE = 100000 # Dataset has over 1 mil values, so we limit to last x values
# SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value # The current symbol to train the model to base predictions on

# TODO: I don't think we need future period, max dataset size, seq_len, symbol_to_predict, dataset. This is the job of DataPreprocesser 
class Model():
    def __init__(self, train_x: np.array, train_y: np.array,
        validation_x: np.array, validation_y: np.array,
        test_x: np.array, test_y: np.array, seq_info:str,
        *,
        max_epochs = 100, batch_size = 1024, hidden_layers = 2,
        neurons_per_layer = 64, architecture: Architecture = Architecture.LSTM.value, dropout = 0.1,
        is_bidirectional = False, initial_learn_rate = 0.001, early_stop_patience = 6):

        ## Param member vars
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.architecture = architecture
        self.dropout = dropout
        self.is_bidirectional = is_bidirectional
        self.initial_learn_rate = initial_learn_rate
        self.seq_info = seq_info

        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_x = test_x
        self.test_y = test_y

        ## Other member vars
        self.model = Sequential()
        self.training_history = None
        self.score = []

        self.create_model()
        
        
    def get_model(self):
        return self.model

    def create_model(self):
        """
        Creates and compiles the model
        """

        ##### Create the model ####
        self.model = Sequential()
        
        self.model.add(self.architecture(self.neurons_per_layer, input_shape=(self.train_x.shape[1:]), return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(BatchNormalization())

        
        for i in range(self.hidden_layers):
            return_sequences = i != self.hidden_layers - 1 # False on last iter
            self.model.add(self.architecture(self.neurons_per_layer, return_sequences=return_sequences))
            self.model.add(Dropout(self.dropout))
            self.model.add(BatchNormalization())

        self.model.add(Dense(1))

        adam = tf.keras.optimizers.Adam(learning_rate=self.initial_learn_rate)
        # Compile model
        self.model.compile(
            loss='mae',
            optimizer=adam,
            metrics=["mae", RSquaredMetric]

        )
        self.save_model_config()


    def train(self):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', early_stop_patience=6)
        tensorboard = TensorBoard(log_dir=f"logs/{self.seq_info}__{self.get_model_info_str()}__{datetime.now().timestamp()}")

        # Train model
        self.training_history = self.model.fit(
            self.train_x, self.train_y,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=(self.validation_x, self.validation_y),
            callbacks=[tensorboard, early_stop],
        )

        # Score model
        self.score = self.model.evaluate(self.validation_x, self.validation_y, verbose=0)
        print('Scores:', self.score)
        # Save model
        self.model.save_weights(f"models/final/{self.seq_info}__{self.get_model_info_str()}__{self.max_epochs}-{self.score[2]:.3f}.h5")

    def save_model(self):
        self.save_model_config()
        self.save_model_weights()

    def save_model_weights(self):
        self.model.save_weights(f"models/final/{self.seq_info}__{self.get_model_info_str()}__{self.max_epochs}-{self.score[2]:.3f}.h5")

    def get_model_info_str(self):
        return f"{self.architecture.__name__}-HidLayers{self.hidden_layers}-Neurons{self.neurons_per_layer}""

    def save_model_config(self):
        json_config = self.model.to_json()
        with open(f'{os.environ["WORKSPACE"]}/model_config/{self.get_model_info_str()}.json', "w+") as file:
            file.write(json_config)

    
