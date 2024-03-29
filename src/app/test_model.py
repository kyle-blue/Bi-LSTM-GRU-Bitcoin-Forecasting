from datetime import datetime, timedelta
import itertools
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import string
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.generic_utils import skip_failed_serialization

# Load test data
# Load model
# Get predictions
# Calculate R Square and MAE (and standard deviation of the errors?)
# Run predictions through simulator
# actual vs predicted plot (for this, dont convert and just show a sample)

def get_model_path():
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
    return chosen_path



def load_model(model_path: str):
    info_list = os.path.split(model_path)[1].split("__")
    model_info = info_list[1]

    model_config_path = f'{os.environ["WORKSPACE"]}/model_config/{model_info}.json'
    model_config = "" 
    
    with open(model_config_path, 'r') as file:
        model_config = file.read()
    model: Sequential = model_from_json(model_config)
    model.load_weights(model_path)
    print(f"Successfully loaded model config from {model_config_path} and weights from {model_path}")
    return model

def get_test_data(model_path: str, is_classification=False):
    info_list = os.path.split(model_path)[1].split("__")
    seq_info = info_list[0]

    print("Loading some unseen test data...")
    state_folder = f'{os.environ["WORKSPACE"]}/state/{seq_info}'
    test_x = np.load(f"{state_folder}/test_x.npy")
    test_y = np.load(f"{state_folder}/test_y.npy")
    percentages = np.array([])
    if is_classification:
        percentages = np.load(f"{state_folder}/percentages.npy")

    return test_x, test_y, percentages

def get_accuracy(predictions: np.ndarray, actual: np.ndarray, percentile = 100.0):
    num_correct = 0
    iterations = 0
    difs = [abs(x - y) for x, y in predictions]
    min_dif = np.percentile(difs, 100.0 - percentile)
    for index, confidences in enumerate(predictions):
        # down_confidence = confidences[0]
        # up_confidence = confidences[1]
        prediction = np.argmax(confidences)
        # print(confidences)
        dif = abs(confidences[0] - confidences[1])
        if dif >= min_dif:
            if prediction == actual[index]:
                num_correct += 1
            iterations += 1
    accuracy = num_correct / iterations
    return accuracy

class SeqInfo():
    symbol: str
    is_classification: bool
    length: int
    forecast_period: int
    
def get_sequence_info(model_path: str):
    info_list = os.path.split(model_path)[1].split("__")
    seq_str = info_list[0]
    seq_str_cap = seq_str.upper()
    seq_info = seq_str.split('-')

    ret = SeqInfo()
    ret.symbol = f"{seq_info[0]}-{seq_info[1]}"
    ret.is_classification = ("CLASS" in seq_str_cap)

    term = "SEQLEN"
    term_loc = seq_str_cap.find(term)
    hyphen_loc = seq_str.find('-', term_loc)
    hyphen_loc = None if hyphen_loc == -1 else hyphen_loc
    ret.length = int(seq_str[term_loc + len(term): hyphen_loc])

    term = "FORWARD"
    term_loc = seq_str_cap.find(term)
    hyphen_loc = seq_str.find('-', term_loc)
    hyphen_loc = None if hyphen_loc == -1 else hyphen_loc
    ret.forecast_period = int(seq_str[term_loc + len(term): hyphen_loc])

    return ret


def get_reg_accuracy(predictions: np.ndarray, actual: np.ndarray, percentile = 100.0):
    upper = np.percentile(predictions, 100.0 - percentile / 2)
    lower = np.percentile(predictions, percentile / 2)

    num_correct = 0
    iterations = 0
    for index, prediction_vec in enumerate(predictions):
        prediction = prediction_vec[0]
        if prediction > upper or prediction < lower:
            if (prediction > 0 and actual[index] > 0) or (prediction < 0 and actual[index] < 0) :
                num_correct += 1
            iterations += 1
    accuracy = num_correct / iterations
    return accuracy

def do_reg_simulation(predictions: np.ndarray, test_y: np.ndarray, forecast_period: int, percentile = 100.0):
    balance = 10000.0
    balances = [balance]
    wins, losses = [], []
    spread = 0.000122 # As fraction of price
    leverage = 2.0
    upper = np.percentile(predictions, 100.0 - percentile / 2)
    lower = np.percentile(predictions, percentile / 2)
    last_trade_index = -np.inf


    print(f"\n\nStart Balance: {balance}")
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        actual = test_y[index]
        if prediction > upper or prediction < lower:
            spread_cost = leverage * balance * spread
            should_buy = prediction > 0 # Buy if positive, sell if negative
            can_trade = index > last_trade_index + forecast_period # Cant trade if already in a trade
            if not can_trade:
                continue
            if should_buy:
                new_balance = balance * (1 + (leverage * actual)) - spread_cost
            else: # We are selling
                new_balance = balance * (1 - (leverage * actual)) - spread_cost
            if new_balance > balance:
                wins.append(abs(leverage * actual))
            else:
                losses.append(abs(leverage * actual))
            last_trade_index = index
            balance = new_balance
            balances.append(balance)

    print(f"Final Balance: {balance: .2f}")
    print("Showing plot for final balance:")
    print(f"{len(balances)} trades executed")
    print(f"{len(predictions)} total predictions")
    print(f"Percentage wins: {len(wins) / (len(balances) - 1) * 100: .2f}%")
    print(f"Average win % of acc: {np.average(wins): .4f}")
    print(f"Average loss % of acc: {np.average(losses): .4f}")
    print(f"Wins: {len(wins)} --- Losses: {len(losses)} --- {len(wins)/(len(wins) + len(losses)) * 100: .2f}% wins")
    print(f"% Predictions < 0: {len(predictions[predictions < 0]) / len(predictions) * 100: .4f}%")

    plt.plot(balances)
    plt.xlabel("Trade Number")
    plt.ylabel("Balance (£)")
    plt.title("Balance Over Simulated Trades")
    plt.draw()


def get_stats(predictions: np.ndarray, actual: np.ndarray, is_classification: bool, percentile = 100.0):
    stats = {}
    
    
    if is_classification:

        difs = np.array([abs(x - y) for x, y in predictions])
        difs1 = np.array([[abs(x - y),abs(x - y)] for x, y in predictions])
        min_dif = np.percentile(difs, 100.0 - percentile)
        act, pred = [], []
        for i in range(len(predictions)):
            if abs(predictions[i][0] - predictions[i][1]) > min_dif:
                act.append(actual[i])
                pred.append(predictions[i])
        act, pred = np.array(act), np.array(pred)
        stats["accuracy"] = get_accuracy(pred, act)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        stats["sparse_categorical_entropy"] = float(scce(act, pred).numpy())

    else:
        upper = np.percentile(predictions, 100.0 - percentile / 2)
        lower = np.percentile(predictions, percentile / 2)
        
        actual = np.ma.masked_where(predictions.flatten() >= upper or predictions.flatten() <= lower, actual)
        predictions = np.ma.masked_where(predictions >= upper or predictions <= lower, predictions)
        
        pred = predictions.flatten()
        stats["mae"] = (np.absolute(pred - actual)).mean()
        correlation_matrix = np.corrcoef(pred, actual)
        correlation_xy = correlation_matrix[0,1]
        stats["r_squared"] = correlation_xy**2
        stats["accuracy"] = get_reg_accuracy(predictions, actual)
    return stats


def plot_predicted(predictions: np.ndarray, actual: np.ndarray, length: int):
    # Plot predicted vs actual
    plt.figure(figsize=(5,2.5))
    plt.plot(predictions.flatten()[:length])
    plt.plot(actual[:length])
    plt.title("Snippet of predicted vs actual prices")
    plt.legend(["Predicted","Actual"])
    plt.xlabel("Prediction number")
    plt.ylabel("Price (% Change)")
    plt.draw()
    plt.show()

def show_confusion_matrix(predictions: np.ndarray, actual: np.ndarray, percentile = 100.0):
    difs = [abs(x - y) for x, y in predictions]
    min_dif = np.percentile(difs, 100.0 - percentile)

    test_pred, test_actual = [], []
    for index, confidences in enumerate(predictions):
        prediction = np.argmax(confidences)
        dif = abs(confidences[0] - confidences[1])
        if dif >= min_dif:
            test_pred.append(prediction)
            test_actual.append(actual[index])

    class_names = ["Down", "Up"]
    cm = confusion_matrix(test_actual, test_pred)

    figure = plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    additional_text = f" at \n{percentile}th percentile" if percentile != 100.0 else ""
    plt.title(f"Confusion matrix{additional_text}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    avg = np.average(cm)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > avg else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.draw()
    plt.show()


def do_simulation(predictions: np.ndarray, test_y: np.ndarray, percentages: np.ndarray, forecast_period = 0, percentile = 0.0):
    balance = 10000.0
    balances = [balance]
    wins, losses = [], []
    spread = 0.000122 # As fraction of price
    leverage = 2.0
    last_trade_index = -np.inf

    difs = [abs(x - y) for x, y in predictions]
    min_dif = np.percentile(difs, 100.0 - percentile)

    print(f"\n\nStart Balance: {balance}")
    for index, confidences in enumerate(predictions):
        prediction = np.argmax(confidences)
        confidence = abs(confidences[prediction])
        actual = test_y[index]
        percent = percentages[index]
        dif = abs(confidences[0] - confidences[1])
        can_trade = index > last_trade_index + forecast_period # Cant trade if already in a trade
        if dif >= min_dif and can_trade:
            spread_cost = leverage * balance * spread
            new_balance = 0
            if prediction == actual: ## Correct direction
                new_balance = balance + (balance * abs(percent) * leverage) - spread_cost
                wins.append(abs(percent) * leverage)
            else: ## Incorrect direction
                new_balance = balance - (balance * abs(percent) * leverage) - spread_cost
                losses.append(abs(percent) * leverage)
            last_trade_index = index
            balance = new_balance
            balances.append(balance)

    print(f"Final Balance: {balance: .2f}")
    print("Showing plot for final balance:")
    print(f"{len(balances)} trades executed")
    print(f"{len(predictions)} total predictions")
    print(f"Percentage wins: {len(wins) / (len(balances) - 1) * 100: .2f}%")
    print(f"Average win % of acc: {np.average(wins) * 100: .2f}%")
    print(f"Average loss % of acc: {np.average(losses) * 100: .2f}%")
    print(f"Wins: {len(wins)} --- Losses: {len(losses)} --- {len(wins)/(len(wins) + len(losses)) * 100: .2f}% wins")

    plt.plot(balances)
    plt.xlabel("Trade Number")
    plt.ylabel("Balance (£)")
    plt.title("Balance Over Simulated Trades")
    plt.draw()

def test_model():
    
    model_path = get_model_path()
    model = load_model(model_path)
    seq_info = get_sequence_info(model_path)
    test_x, test_y, percentages = get_test_data(model_path, seq_info.is_classification)
    predictions = model.predict(test_x)

    stats = get_stats(predictions, test_y, seq_info.is_classification)
    print(f"\n(Test Data) Stats:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    percentile = 20
    stats = get_stats(predictions, test_y, seq_info.is_classification, percentile)
    print(f"\n(Test Data) Stats at {percentile}th percentile:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    if seq_info.is_classification:
        print(f"Accuracy at {percentile} percentile is {get_accuracy(predictions, test_y, percentile)}")
    else:
        print(f"Accuracy at {percentile} percentile is {get_reg_accuracy(predictions, test_y, percentile)}")


    
    if seq_info.is_classification:
        # do_simulation(predictions, test_y)
        do_simulation(predictions, test_y, percentages, seq_info.forecast_period, percentile)
        show_confusion_matrix(predictions, test_y)
        show_confusion_matrix(predictions, test_y, percentile)
    else:
        # do_reg_simulation(predictions, test_y)
        do_reg_simulation(predictions, test_y, percentile)
        plot_predicted(predictions, test_y, length=50)


    

    
