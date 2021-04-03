from enum import Enum
from tensorflow.python.keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU


### SEQ INFO


class Dataset (Enum):
    WEATHER = "WEATHER"
    STOCKS = "STOCKS"


class Symbol (Enum):
    TSLA = "TSLA"
    QQQ = "QQQ"
    SPY = "SPY"
    VXX = "VXX"

class Model (Enum):
    LSTM = LSTM
    CuDNNLSTM = CuDNNLSTM
    GRU = GRU
    CuDNNGRU = CuDNNGRU


DATASET = Dataset.WEATHER.value
SYMBOL_TO_PREDICT = Symbol.TSLA.value # The current symbol to train the model to base predictions on
FUTURE_PERIOD = 25 # The look forward period for the future column, used to train the neural network to predict future price
SEQUENCE_LEN = 120 # The look back period aka the sequence length. e.g if this is 100, the last 100 prices will be used to predict future price

EPOCHS = 100 # Epochs per training fold (we are doing 10 fold cross validation)
BATCH_SIZE = 2048

## MODEL INFO
HIDDEN_LAYERS = 4
NEURONS_PER_LAYER = 64
MODEL = Model.CuDNNGRU
SHOULD_USE_DROPOUT = False