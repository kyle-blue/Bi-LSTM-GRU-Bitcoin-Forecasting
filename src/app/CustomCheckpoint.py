from math import inf
import tensorflow as tf

class CustomCheckPoint(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_mae = inf
        super().__init__()
    
    def on_epoch_end(self, epoch, logs):
        print("Converting CuDNNLSTM into LSTM then saving model (if val_mae improved)")
        mae: float = logs["val_mean_absolute_error"]
        if mae < self.best_mae:
            self.best_mae = mae
        
        return super().on_epoch_end(epoch, logs=logs)