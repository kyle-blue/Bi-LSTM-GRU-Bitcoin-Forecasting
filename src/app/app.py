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

def start():
    print("\n\n\n")

    folder = os.environ["WORKSPACE"]
    dataset_labels = pd.read_csv(f"{folder}/data/hourly/aapl_intraday-60min_historical-data-01-27-2021.csv", usecols=[0]).to_numpy()
    dataset_values = pd.read_csv(f"{folder}/data/hourly/aapl_intraday-60min_historical-data-01-27-2021.csv", usecols=[1]).to_numpy()
    dataset_values = normalize(dataset_values)
    dataset = tf.data.Dataset.from_tensor_slices((dataset_values, dataset_labels))

    test_set = dataset.take(1000)
    train_set = dataset.skip(1000)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(4))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_set, epochs=100, verbose=2)
    # plt.plot(dataset)
    # plt.show()