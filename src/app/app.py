from datetime import datetime
from typing import Deque
from numpy.core.numeric import NaN
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


def normalize(arr: np.array):
    max_val = np.max(arr)
    min_val = np.min(arr)
    max_val -= min_val
    for x in np.nditer(arr, op_flags=['readwrite']):
        x[...] = x - min_val
        x[...] = x / max_val
    return arr


def get_main_dataframe():
    PICKLE_NAME = "data.pkl"
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



def add_target(df: pd.DataFrame):
    ## Add abs_avg column (avg percent change over last x values)
    ## The abs_avg is the absolute average % change over last x values
    avgs = []
    symbol_data = df[f"{SYMBOL_TO_PREDICT}_%_chg"]
    symbol_data_len = len(symbol_data)
    for i in range(symbol_data_len):
        if i < SEQUENCE_LEN:
            avgs.append(NaN)
            continue
        avgs.append(sum([abs(x) for x in symbol_data[i - SEQUENCE_LEN: i]]) / SEQUENCE_LEN)        

    df["abs_avg"] = avgs
    df.dropna(inplace=True)


    ## Add future price column to main_df
    future = []
    symbol_data = df[f"{SYMBOL_TO_PREDICT}_%_chg"]
    symbol_data_len = len(symbol_data)
    for i in range(symbol_data_len):
        if i >= symbol_data_len - FUTURE_PERIOD:
            future.append(NaN)
            continue
        future.append(sum(symbol_data[i:i + FUTURE_PERIOD])) # Add the sum of the last x % changes

    df["future"] = future
    df.dropna(inplace=True)

    ## Add target column to main_df
    targets = []
    for i in range(len(df)):
        multiplier_requirement = FUTURE_PERIOD * ACTION_REQUIREMENT
        if abs(df["future"][i]) > (multiplier_requirement * df["abs_avg"][i]):
            targets.append(1 if df["future"][i] >= 0 else 2) # Long trade target if positive, short trade target of negative
        else:
            targets.append(0)
    df["target"] = targets
    df.dropna(inplace=True)

    ## Remove future price and abs_avg column, we don't need them anymore
    df.drop(columns=["future", "abs_avg"], inplace=True)

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


    ##### PREPROCESSING #####
    for col in df.columns:
        if "volume" in col: 
            ## Change volumes into percent change also
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
        if col != "target":
            ## Normalise all data (except target price)
            df[col] = normalize(df[col].values)
    

    random.shuffle(sequences) # Shuffle sequences to avoid order effects on learning

    # TODO: May have to change the way target is calculated to make it more balanced.
    ##### BALANCING
    buys, sells, none = [], [], []
    for seq, target in sequences:
        if target == 0:
            none.append([seq, target])
        if target == 1:
            buys.append([seq, target])
        if target == 2:
            sells.append([seq, target])
    
    min_values = min(len(buys), len(sells), len(none))
    buys = buys[:min_values]
    sells = sells[:min_values]
    none = none[:min_values]

    sequences = buys + sells + none
    random.shuffle(sequences)

    train_x = []
    train_y = []
    for seq, target in sequences:
        train_x.append(seq)
        train_y.append(target)
    
    train_x = np.array(train_x)
    return train_x, train_y


def get_datasets(df: pd.DataFrame):
    ## Split validation and training set
    times = sorted(df.index.values)
    last_slice = sorted(df.index.values)[-int(0.1*len(times))]

    validation_df = df[(df.index >= last_slice)]
    df = df[(df.index < last_slice)]

    train_x, train_y = preprocess_df(df)
    validation_x, validation_y = preprocess_df(validation_df)

    print(f"\n\nMAIN DF FOR {SYMBOL_TO_PREDICT}")
    print(df.head(15))

    return train_x, train_y, validation_x, validation_y




FUTURE_PERIOD = 5 # The look forward period for the future column, used to train the neural network to predict future price
SEQUENCE_LEN = 150 # The look back period aka the sequence length. e.g if this is 100, the last 100 prices will be used to predict future price
SYMBOL_TO_PREDICT = "TSLA" # The current symbol to train the model to base predictions on
# The requirement for action (long or short) to be taken on a trade.
# e.g. if this is 0.75, and the FUTURE_PERIOD is 4, 0.75*4 is 3. So the future % change must be 3* the avg abs % change to consider a long or short in the target column
# The higher the number, the less trades (and less lenient), the lower the number, the more trades (and more lenient)
ACTION_REQUIREMENT = 0.75
EPOCHS = 50
BATCH_SIZE = 64
NAME = f"{SYMBOL_TO_PREDICT}_5min-SEQ_{SEQUENCE_LEN}-E_{EPOCHS}-F{FUTURE_PERIOD}-A_{ACTION_REQUIREMENT}-v1-{int(time.time())}"

def start():
    print("\n\n\n")

    main_df = get_main_dataframe()

    # Add day column to dataframe
    days, hours = [], []
    for time in main_df.index:
        days.append(float(time.weekday()))
        hours.append(float(time.hour))
    main_df["day"] = days
    main_df["hour"] = hours
    
    main_df = add_target(main_df)

    train_x, train_y, validation_x, validation_y = get_datasets(main_df)

    print(f"Training - Number of buys: {train_y.count(1)}")
    print(f"Training - Number of sells: {train_y.count(2)}")
    print(f"Training - Number of none: {train_y.count(0)}")

    print(f"Validation - Number of buys: {validation_y.count(1)}")
    print(f"Validation - Number of sells: {validation_y.count(2)}")
    print(f"Validation - Number of none: {validation_y.count(0)}")


    ##### Compile / Train the model ###
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(LSTM(64))
    model.add(BatchNormalization())

    # model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

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
    model.save(f"models/{NAME}")

