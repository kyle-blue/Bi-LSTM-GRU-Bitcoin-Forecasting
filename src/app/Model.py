from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU, CuDNNLSTM, CuDNNGRU, Bidirectional
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.client import device_lib
import numpy as np
import os

from app.parameters import Architecture
from .RSquaredMetric import RSquaredMetric, RSquaredMetricNeg


class Model():
    def __init__(self, train_x, train_y,
        validation_x , validation_y, seq_info:str,
        *,
        max_epochs = 100, batch_size = 1024, hidden_layers = 2,
        neurons_per_layer = 64, architecture = Architecture.LSTM.value, dropout = 0.1,
        is_bidirectional = False, initial_learn_rate = 0.001, early_stop_patience = 6,
        random_seed=None):
        """
        INFO GOES HERE
        """

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
        self.early_stop_patience = early_stop_patience
        self.random_seed = random_seed

        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y

        ## Other member vars
        self.model = Sequential()
        self.training_history = None
        self.score: dict = {}

        self._create_model()
        
        
    ### PUBLIC FUNCTIONS

    def get_model(self):
        return self.model

    def train(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=self.early_stop_patience, restore_best_weights=True)
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
        self.score = {out: self.score[i] for i, out in enumerate(self.model.metrics_names)}
        print('Scores:', self.score)

    def save_model(self):
        self._save_model_config()
        self._save_model_weights()

    def get_model_info_str(self):
        return f"{'Bi' if self.is_bidirectional else ''}{self.architecture.__name__}-HidLayers{self.hidden_layers}-Neurons{self.neurons_per_layer}-Bat{self.batch_size}-Drop{self.dropout}"

    ### PRIVATE FUNCTIONS

    def _create_model(self):
        """
        Creates and compiles the model
        """
        self._use_gpu_if_available()

        ##### Create the model ####
        self.model = Sequential()
        
        if self.is_bidirectional:
            self.model.add(Bidirectional(self.architecture(self.neurons_per_layer, input_shape=(self.train_x.shape[1:]), return_sequences=True)))
        else:
            self.model.add(self.architecture(self.neurons_per_layer, input_shape=(self.train_x.shape[1:]), return_sequences=True))
        self.model.add(Dropout(self.dropout, seed=self.random_seed))
        self.model.add(BatchNormalization())

        
        for i in range(self.hidden_layers):
            return_sequences = i != self.hidden_layers - 1 # False on last iter
            if self.is_bidirectional:
                self.model.add(Bidirectional(self.architecture(self.neurons_per_layer, return_sequences=return_sequences)))
            else:
                self.model.add(self.architecture(self.neurons_per_layer, return_sequences=return_sequences))
            self.model.add(Dropout(self.dropout, seed=self.random_seed))
            self.model.add(BatchNormalization())
            

        self.model.add(Dense(1))

        adam = tf.keras.optimizers.Adam(learning_rate=self.initial_learn_rate)


        
        # Compile model
        self.model.compile(
            loss=RSquaredMetricNeg, # Negative to make it maximise RSquared
            optimizer=adam,
            metrics=["mae", RSquaredMetric]
        )


    def _use_gpu_if_available(self):
        ## Utilise GPU if GPU is available
        local_devices = device_lib.list_local_devices()
        gpus = [x.name for x in local_devices if x.device_type == 'GPU']
        if len(gpus) != 0:
            if self.architecture == GRU:
                self.architecture = CuDNNGRU
            elif self.architecture == LSTM:
                self.architecture = CuDNNLSTM

    
    def _save_model_weights(self):
        file_path = f"models/final/{self.seq_info}__{self.get_model_info_str()}__{self.max_epochs}-{self.score['RSquaredMetric']:.3f}.h5"
        self.model.save_weights(file_path)
        print(f"Saved model weights to: {file_path}")

    def _save_model_config(self):
        json_config = self.model.to_json()
        file_path = f'{os.environ["WORKSPACE"]}/model_config/{self.get_model_info_str()}.json'
        with open(file_path, "w+") as file:
            file.write(json_config)
        print(f"Saved model config to: {file_path}")

    
