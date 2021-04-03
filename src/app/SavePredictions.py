import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import tensorflow.python.keras.backend as K
import numpy as np
import time

def current_milli_time():
    return round(time.time() * 1000)


class SavePrediction(Callback):
    def __init__(self, true):
        super().__init__()
        self._get_pred = None
        self.preds = None
        self.true = true
        self.last_time = current_milli_time()

    def _pred_callback(self, preds):
        if self.preds is None:
            self.preds = preds
        else:
            self.preds = np.concatenate((self.preds, preds))

    def set_model(self, model):
        super().set_model(model)
        if self._get_pred is None:
            self._get_pred = self.model.outputs[0]

    def on_test_begin(self, logs):
        # pylint: disable=protected-access
        self.model._make_test_function()
        self.model._make_train_function()
        # pylint: enable=protected-access
        if self._get_pred not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self._get_pred)
            self.model.test_function.fetch_callbacks[self._get_pred] = self._pred_callback
            self.model.train_function.fetches.append(self._get_pred)
            self.model.train_function.fetch_callbacks[self._get_pred] = self._pred_callback

    def on_test_end(self, logs):
        if self._get_pred in self.model.test_function.fetches:
            self.model.test_function.fetches.remove(self._get_pred)
            self.model.train_function.fetches.remove(self._get_pred)
        if self._get_pred in self.model.test_function.fetch_callbacks:
            self.model.test_function.fetch_callbacks.pop(self._get_pred)
            self.model.train_function.fetch_callbacks.pop(self._get_pred)


    def on_epoch_end(self, epoch, logs):
        time = current_milli_time()
        time_elapsed = time - self.last_time
        self.last_time = time
        upper = np.percentile(self.preds, 90)
        lower = np.percentile(self.preds, 10)
        # upper_mask = self.preds > upper
        # lower_mask = self.preds < lower
        # mask = upper_mask or lower_mask
        # extremes = self.preds[mask]


        extreme_mae = 0
        mae = 0
        for index, pred in enumerate(self.preds):
            true = self.true[index]
            pred = pred[0]
            mae += abs(true - pred)
        mae = mae / len(self.preds)

        count = 0
        for index, pred in enumerate(self.preds):
            if pred > upper or pred < lower:
                count += 1
                true = self.true[index]
                pred = pred[0]
                extreme_mae += abs(true - pred)
        extreme_mae = extreme_mae / count

        print(f"MAE: {mae: .4f}")
        print(f"EXTREME_MAE: {extreme_mae: .4f}")
        print(f"HIGHEST PRED: {np.max(self.preds): .4f}")
        print(f"LOWEST PRED: {np.min(self.preds): .4f}")
        print(f"STDEV: {np.std(self.preds): .4f}")
        time_elapsed_seconds = time_elapsed / 1000.0
        print(f"TIME SINCE LAST EPOCH: {time_elapsed_seconds: .3f} seconds")

        self.preds = None
        return super().on_epoch_end(epoch, logs=logs)