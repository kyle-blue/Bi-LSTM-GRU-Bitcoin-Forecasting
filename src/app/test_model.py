import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential
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
    model_config_name = chosen_file.split("__")[1];
    
    model_config_path = f'{os.environ["WORKSPACE"]}/model_config/{model_config_name}.json'
    model_config = "" 
    with open(model_config_path, 'r') as file:
        model_config = file.read()
    model: Sequential = keras.models.model_from_json(model_config)
    model.load_weights(chosen_path)
    print(f"Successfully loaded model config from {model_config_path} and weights from {chosen_path}")

    print("Loading some validation data...")
    STATE_FOLDER = f'{os.environ["WORKSPACE"]}/state/data'
    validation_x = np.load(f"{STATE_FOLDER}/validation_x.npy")
    validation_y = np.load(f"{STATE_FOLDER}/validation_y.npy")

    # ITERATIONS = 100
    # print(f"Testing {ITERATIONS} random sequences for {SYMBOL}")
    predictions = model.predict(validation_x)
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        actual = validation_y[index]
        print(f"Prediction: {prediction} --- actual: {actual}")
    # print(predictions[0][0])


    balance = 10000
    THRESHOLD = 4
    print(f"\n\nStart Balance: {balance}")
    for index, prediction in enumerate(predictions):
        prediction = prediction[0]
        actual = validation_y[index]
        if abs(prediction) > THRESHOLD:
            should_buy = prediction > 0 # Buy if positive, sell if negative
            if should_buy:
                balance = balance * ((100 + actual) / 100)
            else: # We are selling
                balance = balance * ((100 - actual) / 100)
            print(f"Prediction: {prediction} --- Actual price dif: {actual} --- New Bal: {balance}")



    print(f"Final Balance: {balance}")




