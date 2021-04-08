import tensorflow as tf
import os
from app.DataPreprocesser import DataPreprocesser
from app.Model import Model
from app.parameters import Architecture, Symbol
from app.test_model import SYMBOL, test_model
import ta
import matplotlib.pyplot as plt
import numpy as np


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


SYMBOL_TO_PREDICT = Symbol.BTC_USDT.value

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
    print("3. Test indicator correlations")
    is_valid = False
    while not is_valid:
        inp = int(input())
        if inp == 1:
            train_model()
            is_valid = True
        if inp == 2:
            test_model()
            is_valid = True
        if inp == 3:
            indicator_correlations()
            is_valid = True
        if not is_valid:
            print("Please choose a valid option...")

    

def train_model():
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{SYMBOL_TO_PREDICT}.parquet",
        sequence_length=250
    )
    preprocessor.preprocess()

    train_x, train_y = preprocessor.get_train()
    validation_x, validation_y = preprocessor.get_validation()

    model = Model(
        train_x, train_y, validation_x, validation_y,
        preprocessor.get_seq_info_str(),
        architecture=Architecture.LSTM.value,
        is_bidirectional=True
    )
    
    preprocessor.print_dataset_totals()
    del preprocessor # Save memory

    model.train()
    model.save_model()


def indicator_correlations():
    max_df_len = 1000
    preprocessor = DataPreprocesser(
        f"{os.environ['WORKSPACE']}/data/crypto",
        col_names=["open", "high", "low", "close", "volume"],
        forecast_col_name="close",
        forecast_file=f"{SYMBOL_TO_PREDICT}.parquet",
        max_dataset_size=max_df_len,
        should_ask_load=False # Don't load previously generated sequences (and don't ask)
    )
    # We don't need to preprocessor.preprocess() since we don't want the sequences

    df = preprocessor.get_df_original()
    for col in df.columns:
        symbol = col.split("_")[0]
        if symbol == SYMBOL_TO_PREDICT:
            df = ta.add_all_ta_features(
                df, f"{SYMBOL_TO_PREDICT}_open", f"{SYMBOL_TO_PREDICT}_high", 
                f"{SYMBOL_TO_PREDICT}_low", f"{SYMBOL_TO_PREDICT}_close",
                f"{SYMBOL_TO_PREDICT}_volume", fillna=True, colprefix=f"{SYMBOL_TO_PREDICT}_ind_"
            )
            df.dropna(inplace=True)

    print("Added all indicators!")

    ## Remove non-indicators
    for col in df.columns:
        is_indicator = "_ind_" in col
        if not is_indicator:
            del df[col]

    print("Removed all non-indicators!")
    print(df)

    correlations = df.corr()
    figure = plt.figure(figsize=(30, 30))
    ax = figure.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations, interpolation="nearest")
    cb = figure.colorbar(cax)
    cb.ax.tick_params(labelsize=30)
    plt.title('Correlation Matrix', fontsize=40)

    ax.set_xticks(list(range(len(df.columns))))
    ax.set_xticklabels(df.columns,fontsize=10)
    ax.set_yticks(list(range(len(df.columns))))
    ax.set_yticklabels(df.columns, fontsize=10)
    plt.xticks(rotation=90)
    
    plt.show()


            # ind = ta.trend.MACD(df[f"{symbol}_close"], fillna=True)
            # df[f"{symbol}_macd_fast"] = ind.macd()
            # df[f"{symbol}_macd_signal"] = ind.macd_signal()
            # df[f"{symbol}_macd_histogram"] = ind.macd_diff()

            # ind = ta.momentum.RSIIndicator(df[f"{symbol}_close"], fillna=True)
            # df[f"{symbol}_rsi"] = ind.rsi()

            # ind = ta.trend.ADXIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
            # df[f"{symbol}_adx"] = ind.adx()
            # df[f"{symbol}_adx_neg"] = ind.adx_neg()
            # df[f"{symbol}_adx_pos"] = ind.adx_pos()

            # ind = ta.volume.AccDistIndexIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], df[f"{symbol}_volume"], fillna=True)
            # df[f"{symbol}_acc_dist"] = ind.acc_dist_index()

            # ind = ta.volatility.AverageTrueRange(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
            # df[f"{symbol}_atr"] = ind.average_true_range()
