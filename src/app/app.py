from numpy.core.numeric import NaN
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def normalize(arr: np.array):
    max_val = np.max(arr)
    for x in np.nditer(arr, op_flags=['readwrite']):
        x[...] = x / max_val
    return arr


def get_main_dataframe():
    PICKLE_NAME = "data.pkl"
    PICKLE_FOLDER = f'{os.environ["WORKSPACE"]}/state/data'

    data_folder = f'{os.environ["WORKSPACE"]}/data/normal_hours/hourly'
    symbols = [x[:-4] for x in os.listdir(data_folder)] # Remove .csv file extension from strings
    

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




FUTURE_PERIOD = 4 # The look forward period for the future column, used to train the neural network to predict future price
SEQUENCE_LEN = 100 # The look back period aka the sequence length. e.g if this is 100, the last 100 prices will be used to predict future price
SYMBOL_TO_PREDICT = "MS" # The current symbol to train the model to base predictions on

def start():
    print("\n\n\n")

    main_df = get_main_dataframe()

    ## Add future price column to main_df
    symbol_data = main_df[f"{SYMBOL_TO_PREDICT}_%_chg"]
    symbol_data_len = len(symbol_data)
    future = []
    for i in range(symbol_data_len):
        if i >= symbol_data_len - FUTURE_PERIOD:
            future.append(NaN)
            continue
        future.append(sum(symbol_data[i:i + FUTURE_PERIOD])) # Add the sum of the last x % changes

    main_df["future"] = future
    main_df.dropna(inplace=True)

    ## Add target column to main_df
    

    print(f"MAIN DF")
    print(main_df)



    # dataset_labels = pd.read_csv(f"{data_folder}/aapl_intraday-60min_historical-data-01-27-2021.csv", usecols=[0])
    # dataset_values = pd.read_csv(f"{data_folder}/aapl_intraday-60min_historical-data-01-27-2021.csv", usecols=[1])





    # dataset_values = normalize(dataset_values)
    # dataset = tf.data.Dataset.from_tensor_slices((dataset_values, dataset_labels))

    # test_set = dataset.take(1000)
    # train_set = dataset.skip(1000)

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.LSTM(4))
    # model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(train_set, epochs=100, verbose=2)
    # plt.plot(dataset)
    # plt.show()