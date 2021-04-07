import tensorflow as tf
import os
from app.DataPreprocesser import DataPreprocesser
from app.Model import Model
from app.parameters import Architecture, Symbol
from app.test_model import test_model


# TODO: Uncomment this part??
# if symbol == SYMBOL_TO_PREDICT:
#     ind = ta.trend.MACD(df[f"{symbol}_close"], fillna=True)
#     df[f"{symbol}_macd_fast"] = ind.macd()
#     df[f"{symbol}_macd_signal"] = ind.macd_signal()
#     df[f"{symbol}_macd_histogram"] = ind.macd_diff()

#     ind = ta.momentum.RSIIndicator(df[f"{symbol}_close"], fillna=True)
#     df[f"{symbol}_rsi"] = ind.rsi()

#     ind = ta.trend.ADXIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
#     df[f"{symbol}_adx"] = ind.adx()
#     df[f"{symbol}_adx_neg"] = ind.adx_neg()
#     df[f"{symbol}_adx_pos"] = ind.adx_pos()

#     ind = ta.volume.AccDistIndexIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], df[f"{symbol}_volume"], fillna=True)
#     df[f"{symbol}_acc_dist"] = ind.acc_dist_index()

#     ind = ta.volatility.AverageTrueRange(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
#     df[f"{symbol}_atr"] = ind.average_true_range()




def init_tf():
    ## Only allocate required GPU space
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    tf.compat.v1.disable_eager_execution()


def start():
    init_tf()

    print("\n\n\n")
    print("Please choose an option:")
    print("1. Train a new model")
    print("2. Test an existing model")
    is_valid = False
    while not is_valid:
        inp = int(input())
        if inp == 1:
            train_model()
            is_valid = True
        if inp == 2:
            test_model()
            is_valid = True
        if not is_valid:
            print("Please choose a valid option...")

    

def train_model():
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{Symbol.BTC_USDT.value}.parquet",
        sequence_length=250
    )
    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()

    model = Model(
        train_x, train_y, validation_x, validation_y,
        preprocessor.get_seq_info_str(),
        architecture=Architecture.LSTM.value
    )
    
    preprocessor.print_dataset_totals()
    del preprocessor # Save memory

    model.train()
    model.save_model()


