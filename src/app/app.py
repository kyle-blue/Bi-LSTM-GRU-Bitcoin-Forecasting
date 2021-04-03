from datetime import datetime
from typing import Deque
from numpy.core.numeric import Inf, NaN
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, BatchNormalization, CuDNNLSTM, CuDNNGRU
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.python.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import os
import random
from app.CustomMetrics import CustomMetrics
from app.RSquaredMetric import RSquaredMetric
from app.SavePredictions import SavePrediction
from app.extreme_mae import extreme_mae
import ta

from app.test_model import test_model



SEQ_INFO = f"{SYMBOL_TO_PREDICT}-{INTERVAL}-SeqLen{SEQUENCE_LEN}-Forward{FUTURE_PERIOD}-{'NoGap' if REMOVE_GAP_UPS else 'Gap'}"
LAYER_NAME = "LSTM" if RNN is CuDNNLSTM else "GRU"
MODEL_INFO = f"{LAYER_NAME}-HidLayers{HIDDEN_LAYERS}-Neurons{NEURONS_PER_LAYER}"

def normalize(arr: np.array):
    return (arr - np.mean(arr)) / np.std(arr)

def get_main_dataframe():
    PICKLE_NAME = "dataframe.pkl"
    PICKLE_FOLDER = f'{os.environ["WORKSPACE"]}/state/{SEQ_INFO}'
    if not os.path.exists(PICKLE_FOLDER):
        os.makedirs(PICKLE_FOLDER)

    data_folder = ""
    if INTERVAL == "1min":
        data_folder = f'{os.environ["WORKSPACE"]}/data/trading/normal_hours/1min'
    if INTERVAL == "5min":
        data_folder = f'{os.environ["WORKSPACE"]}/data/trading/extended_hours/5min'
    symbols = set([x[:-4] for x in os.listdir(data_folder)]) # Remove .csv file extension from strings
    
    
    if PICKLE_NAME in os.listdir(PICKLE_FOLDER):
        main_df = pd.read_pickle(f"{PICKLE_FOLDER}/{PICKLE_NAME}")
        print("Found existing dataframe, checking if it has correct number of columns (2x the number of symbols [1 for %_chg and 1 for volume])")
        print(main_df.tail(15))
        if len(main_df.columns) == 5 * len(symbols) + 9: # There are 9 extra indicators
            print(f"Dataframe found to be the most up to date, using dataframe in: {PICKLE_FOLDER}/{PICKLE_NAME}")
            return main_df

    main_df = pd.DataFrame()
    for symbol in symbols:
        filename = f"{data_folder}/{symbol}.csv"
        df = pd.read_csv(filename, parse_dates=["Time"])
        df.set_index("Time", inplace=True)
        df = df[["Open", "High", "Low", "Last", "Volume"]]
        df = df.iloc[::-1] # Reverse dataset so its earliest to latest

        df.rename(columns={"Open": f"{symbol}_open", "High": f"{symbol}_high", "Low": f"{symbol}_low", "Last": f"{symbol}_close", "Volume": f"{symbol}_volume"}, inplace=True)

        if symbol == SYMBOL_TO_PREDICT:
            ind = ta.trend.MACD(df[f"{symbol}_close"], fillna=True)
            df[f"{symbol}_macd_fast"] = ind.macd()
            df[f"{symbol}_macd_signal"] = ind.macd_signal()
            df[f"{symbol}_macd_histogram"] = ind.macd_diff()

            ind = ta.momentum.RSIIndicator(df[f"{symbol}_close"], fillna=True)
            df[f"{symbol}_rsi"] = ind.rsi()

            ind = ta.trend.ADXIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
            df[f"{symbol}_adx"] = ind.adx()
            df[f"{symbol}_adx_neg"] = ind.adx_neg()
            df[f"{symbol}_adx_pos"] = ind.adx_pos()

            ind = ta.volume.AccDistIndexIndicator(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], df[f"{symbol}_volume"], fillna=True)
            df[f"{symbol}_acc_dist"] = ind.acc_dist_index()

            ind = ta.volatility.AverageTrueRange(df[f"{symbol}_high"], df[f"{symbol}_low"], df[f"{symbol}_close"], fillna=True)
            df[f"{symbol}_atr"] = ind.average_true_range()

        if len(main_df) == 0: main_df = df
        else: main_df = main_df.join(df, how="outer")

        print(main_df.tail(20))
        main_df.dropna(inplace=True)
    pd.to_pickle(main_df, f"{PICKLE_FOLDER}/{PICKLE_NAME}") # Save df to avoid future processing
    return main_df



def add_derived_data(df: pd.DataFrame):
    # Add day column to dataframe
    days, hours, minutes = [], [], []
    for time in df.index:
        days.append(float(time.weekday()))
        # hours.append(float(time.hour))
        minutes.append(float(time.minute) + (float(time.hour) * 60))
    df["day"] = days
    # df["hour"] = hours
    df["minute"] = minutes
    df["minute_not_norm"] = minutes
    

    ##### PREPROCESSING NORMALISATION #####
    for col in df.columns:
        if col != "minute_not_norm":
            ## Change volumes into percent change also
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            ## Normalise all data (except target price)
            df[col] = normalize(df[col].values, col)

    ## Add future price column to main_df (which is now the target)
    future = []
    symbol_data = df[f"{SYMBOL_TO_PREDICT}_close"]
    symbol_data_len = len(symbol_data)
    for i in range(symbol_data_len):
        if i >= symbol_data_len - FUTURE_PERIOD:
            future.append(NaN)
            continue
        future.append(sum(symbol_data[i:i + FUTURE_PERIOD])) # Add the sum of the last x % changes

    df["target"] = future
    df.dropna(inplace=True)

    return df


## @returns train_x and train_y
def preprocess_df(df: pd.DataFrame, isTest = False):
    ## Create sequences
    # [
    #    [[sequence1], target1]
    #    [[sequence2], target2]
    # ]
    
    
    sequences: list = [] 
    cur_sequence: Deque = deque(maxlen=SEQUENCE_LEN)
    # minute_index = df.columns.get_loc("minute")
    minutes = df["minute_not_norm"]
    df.drop(columns=["minute_not_norm"])
    target_index = df.columns.get_loc("target")
    min_minute = minutes.min() # Minute for start of day
    max_minute = minutes.max() # Minute for end of day
    for index, value in enumerate(df.to_numpy()):
        # Since value is only considered a single value in the sequence (even though itself is an array), to make it a sequence, we encapsulate it in an array so:
        # sequence1 = [[values1], [values2], [values3]]
        if REMOVE_GAP_UPS:
            cur = minutes[index]
            if cur >= max_minute - FUTURE_PERIOD:
                continue
            if cur == min_minute:
                cur_sequence.clear() # Only allowed full sequences
        cur_sequence.append(value[:target_index]) # Append all but target to cur_sequence
        if len(cur_sequence) == SEQUENCE_LEN:
            seq = list(cur_sequence)
            seq.reverse()
            sequences.append([np.array(seq), value[target_index]]) # value[-1] is the target
    
    df.drop_duplicates(inplace=True)
    
    if not isTest:
        random.shuffle(sequences) # Shuffle sequences to avoid order effects on learning

    train_x = []
    train_y = []
    for seq, target in sequences:
        train_x.append(seq)
        train_y.append(target)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y


def get_datasets():
    FILE_NAMES = ["train_x.npy", "train_y.npy", "validation_x.npy", "validation_y.npy", "test_x.npy", "test_y.npy"]
    STATE_FOLDER = f'{os.environ["WORKSPACE"]}/state/{SEQ_INFO}'
    if not os.path.exists(STATE_FOLDER):
        os.makedirs(STATE_FOLDER)
    
    ### Check for existing training data
    dir_items = os.listdir(STATE_FOLDER)
    if all([x in dir_items for x in FILE_NAMES]):
        print("\n\nFound an training and validation data. Please select an option:")
        print("1. Use existing data")
        print("2. Generate new data")
        is_valid_input = False
        while not is_valid_input:
            user_input = int(input())
            if user_input == 1:
                is_valid_input = True
                print("Using existing data...")
                train_x = np.load(f"{STATE_FOLDER}/{FILE_NAMES[0]}")
                train_y = np.load(f"{STATE_FOLDER}/{FILE_NAMES[1]}")
                validation_x = np.load(f"{STATE_FOLDER}/{FILE_NAMES[2]}")
                validation_y = np.load(f"{STATE_FOLDER}/{FILE_NAMES[3]}")
                test_x = np.load(f"{STATE_FOLDER}/{FILE_NAMES[3]}")
                test_y = np.load(f"{STATE_FOLDER}/{FILE_NAMES[4]}")
                return train_x, train_y, validation_x, validation_y, test_x, test_y
            if user_input == 2:
                print("Generating new arrays of sequences for training...")
                is_valid_input = True
    
    df = get_main_dataframe()
    df = add_derived_data(df)

    ## Split validation and training set 60% 20% 20%
    df, test_df = np.split(df, [int(0.8 * len(df))])
    df.sample(frac = 1) # Shuffle validation and train together (but not test)
    train_df, validation_df = np.split(df, [int(-len(test_df))]) # Validation is same size as test_df

    ## Preprocess
    train_x, train_y = preprocess_df(train_df)
    validation_x, validation_y = preprocess_df(validation_df)
    test_x, test_y = preprocess_df(test_df, True)

    ## Save data to PKL files
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[0]}", train_x)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[1]}", train_y)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[2]}", validation_x)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[3]}", validation_y)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[4]}", test_x)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[5]}", test_y)

    print(f"\n\nMAIN DF FOR {SYMBOL_TO_PREDICT}")
    print(df.head(15))

    return train_x, train_y, validation_x, validation_y, test_x, test_y






def start():
    ## Only allocate required GPU space
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    tf.compat.v1.disable_eager_execution()


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

    train_x, train_y, validation_x, validation_y, test_x, test_y = get_datasets()

    print(f"Training total: {len(train_y)}")
    print(f"Validation total: {len(validation_y)}")
    print(f"Test total: {len(test_y)}")

    ##### Compile / Train the model ###
    
    model = Sequential()    
    model.add(RNN(NEURONS_PER_LAYER, input_shape=(train_x.shape[1:]), return_sequences=True))
    if SHOULD_USE_DROPOUT:
        model.add(Dropout(0.1))
    model.add(BatchNormalization())

    
    for i in range(HIDDEN_LAYERS):
        return_sequences = i != HIDDEN_LAYERS - 1 # False on last iter
        model.add(RNN(NEURONS_PER_LAYER, return_sequences=return_sequences))
        if SHOULD_USE_DROPOUT:
            model.add(Dropout(0.1))
        model.add(BatchNormalization())

    model.add(Dense(1))


    # opt = tf.keras.optimizers.Adam(lr=0.003, decay=1e-6)



    # Compile model
    model.compile(
        loss='mae',
        optimizer="adam",
        metrics=['mse', "mae", RSquaredMetric]
    )

    json_config = model.to_json()
    with open(f'{os.environ["WORKSPACE"]}/model_config/{MODEL_INFO}.json', "w+") as file:
        file.write(json_config)

    tensorboard = TensorBoard(log_dir=f"logs/{SEQ_INFO}__{MODEL_INFO}__{datetime.now().timestamp()}")

    filepath = f"{SEQ_INFO}__{MODEL_INFO}__" + "{epoch:02d}-{val_mae:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint(f"models/{filepath}.h5", monitor="val_mae", verbose=1, save_best_only=False, mode='min', save_weights_only=True) # saves only the best ones

    
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        # callbacks=[tensorboard, checkpoint, SavePrediction(validation_y)],
        callbacks=[tensorboard, checkpoint],
    )

    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Scores:', score)
    # Save model
    model.save_weights(f"models/final/{SEQ_INFO}__{MODEL_INFO}__{EPOCHS}-{score[2]:.3f}.h5")


