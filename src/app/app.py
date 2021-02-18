from datetime import datetime
from typing import Deque
from numpy.core.numeric import Inf, NaN
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, BatchNormalization, CuDNNLSTM
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import os
import random
import time

from app.test_model import test_model


def normalize(arr: np.array, col: str):
    return (arr - np.mean(arr)) / np.std(arr)

def get_main_dataframe():
    PICKLE_NAME = "dataframe.pkl"
    PICKLE_FOLDER = f'{os.environ["WORKSPACE"]}/state/data'

    data_folder = f'{os.environ["WORKSPACE"]}/data/extended_hours/5min'
    symbols = set([x[:-4] for x in os.listdir(data_folder)]) # Remove .csv file extension from strings
    
    
    if PICKLE_NAME in os.listdir(PICKLE_FOLDER):
        main_df = pd.read_pickle(f"{PICKLE_FOLDER}/{PICKLE_NAME}")
        print("Found existing dataframe, checking if it has correct number of columns (2x the number of symbols [1 for %_chg and 1 for volume])")
        if len(main_df.columns) == 2 * len(symbols):
            print(f"Dataframe found to be the most up to date, using dataframe in: {PICKLE_FOLDER}/{PICKLE_NAME}")
            return main_df

    main_df = pd.DataFrame()
    for symbol in symbols:
        filename = f"{data_folder}/{symbol}.csv"
        df = pd.read_csv(filename, parse_dates=["Time"])
        df.set_index("Time", inplace=True)
        df = df[["%Chg", "Volume"]]

        ## Convert %Chg column from string to float64
        change_floats = [float(x[:-1]) for x in df["%Chg"]]
        df["%Chg"] = change_floats

        df.rename(columns={"%Chg": f"{symbol}_%_chg", "Volume": f"{symbol}_volume"}, inplace=True)
        
        if len(main_df) == 0: main_df = df
        else: main_df = main_df.join(df, how="outer")

        main_df.dropna(inplace=True)
    
    pd.to_pickle(main_df, f"{PICKLE_FOLDER}/{PICKLE_NAME}") # Save df to avoid future processing
    return main_df



def add_derived_data(df: pd.DataFrame):
    # Add day column to dataframe
    days, hours, minutes = [], [], []
    for time in df.index:
        days.append(float(time.weekday()))
        hours.append(float(time.hour))
        minutes.append(float(time.minute))
    df["day"] = days
    df["hour"] = hours
    df["minute"] = hours
    

    ## Add future price column to main_df (which is now the target)
    future = []
    symbol_data = df[f"{SYMBOL_TO_PREDICT}_%_chg"]
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
def preprocess_df(df: pd.DataFrame):
    ## Create sequences
    # [
    #    [[sequence1], target1]
    #    [[sequence2], target2]
    # ]
    
    sequences: list = [] 
    cur_sequence: Deque = deque(maxlen=SEQUENCE_LEN)
    for value in df.to_numpy():
        # Since value is only considered a single value in the sequence (even though itself is an array), to make it a sequence, we encapsulate it in an array so:
        # sequence1 = [[values1], [values2], [values3]] 
        cur_sequence.append(value[:-1]) # Append all but target to cur_sequence
        if len(cur_sequence) == SEQUENCE_LEN:
            sequences.append([np.array(cur_sequence), value[-1]]) # value[-1] is the target
    
    df.drop_duplicates(inplace=True)
    ##### PREPROCESSING #####
    for col in df.columns:
        if "volume" in col: 
            ## Change volumes into percent change also
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
        if col != "target":
            ## Normalise all data (except target price)
            df[col] = normalize(df[col].values, col)
    
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
    FILE_NAMES = ["train_x.npy", "train_y.npy", "validation_x.npy", "validation_y.npy"]
    STATE_FOLDER = f'{os.environ["WORKSPACE"]}/state/data'
    
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
                return train_x, train_y, validation_x, validation_y
            if user_input == 2:
                print("Generating new arrays of sequences for training...")
                is_valid_input = True
    
    df = get_main_dataframe()
    df = add_derived_data(df)

    ## Split validation and training set
    times = sorted(df.index.values)
    last_slice = sorted(df.index.values)[-int(0.2*len(times))]

    validation_df = df[(df.index >= last_slice)]
    df = df[(df.index < last_slice)]

    ## Preprocess
    train_x, train_y = preprocess_df(df)
    validation_x, validation_y = preprocess_df(validation_df)

    ## Save data to PKL files
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[0]}", train_x)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[1]}", train_y)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[2]}", validation_x)
    np.save(f"{STATE_FOLDER}/{FILE_NAMES[3]}", validation_y)

    print(f"\n\nMAIN DF FOR {SYMBOL_TO_PREDICT}")
    print(df.head(15))

    return train_x, train_y, validation_x, validation_y




FUTURE_PERIOD = 50 # The look forward period for the future column, used to train the neural network to predict future price
SEQUENCE_LEN = 300 # The look back period aka the sequence length. e.g if this is 100, the last 100 prices will be used to predict future price
SYMBOL_TO_PREDICT = "TSLA" # The current symbol to train the model to base predictions on
EPOCHS = 20 # Epochs per training fold (we are doing 10 fold cross validation)
BATCH_SIZE = 2048
NAME = f"{SYMBOL_TO_PREDICT}_5min-SEQ_{SEQUENCE_LEN}-E_{EPOCHS}-F{FUTURE_PERIOD}-v1-{int(time.time())}"

def start():
    ## Only allocate required GPU space
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

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

    train_x, train_y, validation_x, validation_y = get_datasets()

    print(f"Training total: {len(train_y)}")
    print(f"Validation total: {len(validation_y)}")

    ##### Compile / Train the model ###
    
    model = Sequential()
    model.add(CuDNNLSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(BatchNormalization())

    HIDDEN_LAYERS = 4
    for i in range(HIDDEN_LAYERS):
        return_sequences = i != HIDDEN_LAYERS - 1 # False on last iter
        model.add(CuDNNLSTM(32, return_sequences=return_sequences))
        model.add(BatchNormalization())

    model.add(Dense(1))


    # opt = tf.keras.optimizers.Adam(lr=0.003, decay=1e-6)

    # Compile model
    model.compile(
        loss='mse',
        optimizer="adam",
        metrics=['mse', "mae"]
    )

    json_config = model.to_json()
    with open(f'{os.environ["WORKSPACE"]}/model_config/model_config.json', "w+") as file:
        file.write(json_config)

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_mae:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint(f"models/{filepath}.model.h5", monitor="val_mae", verbose=1, save_best_only=True, mode='min', save_weights_only=True) # saves only the best ones

    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint],
    )

    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save_weights(f"models/final/{NAME}.h5")


