from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import string

SYMBOL = "TSLA"

def test_model():
    model_dir = f'{os.environ["WORKSPACE"]}/models/final'
    dir_items = os.listdir(model_dir)
    dir_items.remove(".temp")
    print("Please choose an model to test from the list (in dir ./model_config):")

    for index, item in enumerate(dir_items):
        print(f"{index}. {item}")

    is_valid_input = False
    chosen_file = ""
    while not is_valid_input:
        user_input = int(input())
        if user_input < len(dir_items):
            is_valid_input = True
        chosen_file = dir_items[user_input]
        if not is_valid_input:
            print("Please enter a valid input")

    chosen_path = f"{model_dir}/{chosen_file}"
    file_info = chosen_file.split("__");
    seq_info = file_info[0]
    model_info = file_info[1];
    
    model_config_path = f'{os.environ["WORKSPACE"]}/model_config/{model_info}.json'
    model_config = "" 
    with open(model_config_path, 'r') as file:
        model_config = file.read()
    model: Sequential = keras.models.model_from_json(model_config)
    model.load_weights(chosen_path)
    print(f"Successfully loaded model config from {model_config_path} and weights from {chosen_path}")

    print("Loading some unseen test data...")
    STATE_FOLDER = f'{os.environ["WORKSPACE"]}/state/{seq_info}'
    test_x = np.load(f"{STATE_FOLDER}/test_x.npy")
    test_y = np.load(f"{STATE_FOLDER}/test_y.npy")

    # ITERATIONS = 100
    # print(f"Testing {ITERATIONS} random sequences for {SYMBOL}")
    predictions = model.predict(test_x)
    pred = predictions.flatten()
    true = test_y.flatten()
    mae = (np.absolute(pred - true)).mean()
    correlation_matrix = np.corrcoef(pred, true)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(f"mae = {mae}")
    print(f"R Squared = {r_squared}")
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        actual = test_y[index]
        # print(f"Prediction: {prediction} --- actual: {actual}")
    # print(predictions[0][0])


    # balance = 10000.0
    # balances = [balance]
    # wins, losses = [], []
    # risk = 1.0 # In percentage
    # commission = 4.0 # As percentage of risk per trade
    # upper = np.percentile(predictions, 90)
    # lower = np.percentile(predictions, 10)
    # print(f"Upper: {upper} --- Lower: {lower}")

    # print(f"\n\nStart Balance: {balance}")
    num_correct_signs = 0
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
    #     multiplier = risk / abs(prediction)
        actual = test_y[index]
        if (prediction < 0 and actual < 0) or (prediction > 0 and actual > 0):
            num_correct_signs += 1
    #     if prediction > upper or prediction < lower:
    #         com = balance * (risk / 100) * (commission / 100)
    #         should_buy = prediction > 0 # Buy if positive, sell if negative
    #         if should_buy:
    #             new_balance = balance * ((100 + (actual * multiplier)) / 100) - com
    #         else: # We are selling
    #             new_balance = balance * ((100 - (actual * multiplier)) / 100) - com
    #         if new_balance > balance:
    #             wins.append(abs(actual * multiplier))
    #         else:
    #             losses.append(abs(actual * multiplier))
    #         balance = new_balance
    #         balances.append(balance)
    #         prediction_ws = " " if prediction > 0 else "" # Whitespace to align print
    #         actual_ws = " " if actual > 0 else "" # Whitespace to align print
    #         # print(f"Prediction: {prediction_ws}{prediction:.3f} --- Actual price dif: {actual_ws}{actual:.3f} --- New Bal: {balance:.2f}")

    # print(f"Final Balance: {balance: .2f}")
    # print("Showing plot for final balance:")
    # print(f"{len(balances)} trades executed")
    # print(f"{len(predictions)} total predictions")
    print(f"Correct direction predicted {num_correct_signs} out of {len(predictions)} times ({num_correct_signs/len(predictions)*100: .2f}%)")
    # print(f"Average win % of acc: {np.average(wins): .2f}")
    # print(f"Average loss % of acc: {np.average(losses): .2f}")
    # print(f"Wins: {len(wins)} --- Losses: {len(losses)} --- {len(wins)/(len(wins) + len(losses)) * 100: .2f}% wins")
    # print(f"% Predictions < 0: {len(predictions[predictions < 0]) / len(predictions) * 100: .4f}%")

    # plt.plot(balances)
    # plt.yscale("log")
    # plt.title("Balance Over Simulated Trades")
    # plt.draw()


    ## Load original price data and plot
    data_folder = ""
    if "1min" in seq_info:
        data_folder = f'{os.environ["WORKSPACE"]}/data/trading/normal_hours/1min'
    if "5min" in seq_info:
        data_folder = f'{os.environ["WORKSPACE"]}/data/trading/extended_hours/5min'
    

    temp = seq_info.split('-')
    symbol = temp[0]
    look_forward = int(temp[-2].strip(string.ascii_letters))
    sequence_len = int(temp[-3].strip(string.ascii_letters))
    filename = f"{data_folder}/{symbol}.csv"
    df = pd.read_csv(filename, parse_dates=["Time"])
    df.set_index("Time", inplace=True)
    df = df[["Last"]]
    df = df[::-1]

    df.rename(columns={"Last": "close"}, inplace=True)

    
   

    ## Get Last 20% of data (test_data)
    train_df, test_df = np.split(df, [int(0.8 * len(df))])
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)
    train_df = train_df[["close"]]
    df.reset_index(inplace=True)
    df = df[["close"]]
    ## Get absolute start price of last 20%


    pred_df = pd.DataFrame()
       
    # for count in range(len(pred)):
    #     prev_price = df["close"][int(0.8 * len(df)) + count]
    #     prediction = pred[count]
    #     actual = true[count]
    #     new_pred = {"close": prev_price + (prev_price * (prediction / 100))}
    #     new_norm = {"close": prev_price + (prev_price * (actual / 100))}
    #     train_df = train_df.append(new_norm, ignore_index=True)
    #     pred_df = pred_df.append(new_pred, ignore_index=True)
    pred_df = pred_df.append( {"close": test_df["close"][0]}, ignore_index=True)
    true_df = pred_df.append( {"close": test_df["close"][0]}, ignore_index=True)
    
    for count in range(len(pred)):
        prev_pred_price = pred_df["close"][count]
        prev_true_price = true_df["close"][count]
        prediction = pred[count] / look_forward # since look forward is > 1 we must divide
        actual = true[count] / look_forward
        new_pred = {"close": prev_pred_price + (prev_pred_price * (prediction / 100))}
        new_true = {"close": prev_true_price + (prev_true_price * (actual / 100))}
        pred_df = pred_df.append(new_pred, ignore_index=True)
        true_df = true_df.append(new_true, ignore_index=True)

    train_df = train_df.append(true_df, ignore_index=True)

    pred_df.index = range( int(0.8 * len(df)), len(pred_df) + int(0.8 * len(df)) )

    plt.figure(figsize=(40, 15))
    plt.plot(train_df)
    plt.draw()
    ## Plot
    plt.plot(pred_df)
    plt.draw()
    # plt.show()

  
    print(f"Mean Absolute Error = {mae}")
    print(f"R Squared = {r_squared}")
