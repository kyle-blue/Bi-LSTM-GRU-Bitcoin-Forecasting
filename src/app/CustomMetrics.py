import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import tensorflow.python.keras.backend as K

class CustomMetrics(Callback):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.outputs = []
        
        # References to the most recent y_true and y_pred (updated from custom metric)
        self.output = tf.Variable(0.0, shape=tf.TensorShape(None))
        

    def on_batch_end(self, batch, logs):
        """Evaluate the variables and save them into lists."""
        self.outputs.append(K.eval(self.output))
        return super().on_batch_end(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        """Print all variables."""
        print("Targets: ", *self.targets)
        print("Outputs: ", *self.outputs)