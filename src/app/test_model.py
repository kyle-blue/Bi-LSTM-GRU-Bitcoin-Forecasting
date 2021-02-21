import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import numpy as np

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
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        actual = test_y[index]
        print(f"Prediction: {prediction} --- actual: {actual}")
    # print(predictions[0][0])


    balance = 10000.0
    balances = [balance]
    wins, losses = [], []
    risk = 1.0 # In percentage
    commission = 4.0 # As percentage of risk per trade
    upper = np.percentile(predictions, 90)
    lower = np.percentile(predictions, 10)
    print(f"Upper: {upper} --- Lower: {lower}")

    print(f"\n\nStart Balance: {balance}")
    num_correct_signs = 0
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        multiplier = risk / abs(prediction)
        actual = test_y[index]
        if (prediction < 0 and actual < 0) or (prediction > 0 and actual > 0):
            num_correct_signs += 1
        if prediction > upper or prediction < lower:
            com = balance * (risk / 100) * (commission / 100)
            should_buy = prediction > 0 # Buy if positive, sell if negative
            if should_buy:
                new_balance = balance * ((100 + (actual * multiplier)) / 100) - com
            else: # We are selling
                new_balance = balance * ((100 - (actual * multiplier)) / 100) - com
            if new_balance > balance:
                wins.append(abs(actual * multiplier))
            else:
                losses.append(abs(actual * multiplier))
            balance = new_balance
            balances.append(balance)
            prediction_ws = " " if prediction > 0 else "" # Whitespace to align print
            actual_ws = " " if actual > 0 else "" # Whitespace to align print
            print(f"Prediction: {prediction_ws}{prediction:.3f} --- Actual price dif: {actual_ws}{actual:.3f} --- New Bal: {balance:.2f}")

    print(f"Final Balance: {balance}")
    print("Showing plot for final balance:")
    print(f"{len(balances)} trades executed")
    print(f"{len(predictions)} total predictions")
    print(f"Correct direction predicted {num_correct_signs} out of {len(predictions)} times ({num_correct_signs/len(predictions)*100}%)")
    print(f"Average win % of acc: {np.average(wins)}")
    print(f"Average loss % of acc: {np.average(losses)}")
    print(f"Wins: {len(wins)} --- Losses: {len(losses)} --- {len(wins)/(len(wins) + len(losses)) * 100}% wins")

    plt.plot(balances)
    plt.yscale("log")
    plt.title("Balance Over Simulated Trades")



